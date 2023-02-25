import math
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
import seaborn as sn
from pandas import option_context
#from sklearn import metrics

import config
from db_interaction import pull
from db_interaction.entity_gateway import EntityGateway
from evaluation.attribution import Attribution
from evaluation.colorhash import ColorHash
from modeling.qiammm import QIAmmm
from prior_factory.distribution_provider import DistributionProvider
from prior_factory.distribution_provider import Distribution_Type

'''
To do:
* [!] Get rid of try-catch. Catch errors explicitly
* show_prior_vs_posterior_plot only works non-derived priors; marry with show_prior_posterior_plot?
* show_posterior_pred_plot: try to run on linux machine.
* Optional: find other way to implement raftery_lewis: method 'raftery_lewis' seems unavailable in pymc3 3.6
'''


class ResultReporter:

    def __init__(self, model, push_ids=-1):
        pd.set_option('display.max_columns', 7)
        pd.set_option('display.width', 1000)
        self.__model = model
        self.__r2_bayesian = self.__set_r2_bayesian()
        self.__r2_regular = self.__set_r2_regular()
        if not isinstance(push_ids, list) and push_ids != -1:
            raise TypeError("[!] 'push_ids' must be of type list. Input '[push_ids]' instead.")
        if push_ids == -1:
            self.__push_id = push_ids
            self.__waic = self.__set_waic()
            self.__r_hat_max = self.__set_r_hat_max()
        elif len(push_ids) == 1:
            self.__push_id = push_ids[0]
            self.__waic = EntityGateway.get_waic(self.__push_id)
            self.__r_hat_max = -1  # We could pull r_hat_max but we don't need it in the result reporter
        else:
            self.__push_id = '_'.join(push_ids)
            self.__waic = -1
            self.__r_hat_max = -1

    # region properties

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        if not isinstance(model, QIAmmm):
            raise TypeError('Input needs to be of type BayesRegression')
        self.__model = model

    @property
    def waic(self):
        return self.__waic

    @property
    def r2_bayesian(self):
        return self.__r2_bayesian

    @property
    def r2_regular(self):
        return self.__r2_regular

    @property
    def r_hat_max(self):
        return self.__r_hat_max
    # endregion

    # region basics

    def __set_waic(self):
        waic = -1
        try:
            waic = round(pm.waic(self.__model.trace, self.__model.pm_model).WAIC, 2)
        except Exception as err:
            print("[!] Set WAIC to -1 because could not set WAIC due to: {0}".format(err))
        return waic

    def __set_r2_bayesian(self):
        r2_bayesian = -1
        try:
            r2s_bayesian = [pm.r2_score(component.dependent.values, component.result_values)[1] for component in self.__model.component_container]
            r2_bayesian = round(np.mean(r2s_bayesian), 2)
        except Exception as err:
            print("[!] Set Bayesian R^2 to -1 because could not set R^2 due to: {0}".format(err))
        return r2_bayesian

    def __set_r2_regular(self):
        r2_regular = -1
        try:
            obs_pred = self.__get_obs_pred_aggregated_dict()
            r2 = np.corrcoef(obs_pred['observed'], obs_pred['predicted'])[0,1] ** 2
            if isinstance(r2, complex):  # find out why r2 sometimes becomes a complex number. negative values and log?
                 r2 = r2.real  # + r2.imag
            r2_regular = round(r2, 2)
            # r2_regular = round(metrics.r2_score(obs_pred['observed'], obs_pred['predicted']), 2)
        except Exception as err:
            print("[!] Set regular R^2 to -1 because could not set R^2 due to: {0}".format(err))
        return r2_regular

    def __set_r_hat_max(self):
        r_hat_max = -1
        try:
            r_hats = {}
            for metric_key, channels in az.summary(self.__model.trace).items():
                for ch_key, value in channels.items():
                    if metric_key == 'r_hat':
                        r_hats[ch_key] = value
            r_hat_max = max(r_hats.values())
        except Exception as err:
            print("[!] Set rhat_max to -1 because could not set rhat_max due to: {0}".format(err))
        return r_hat_max

    def __set_model_score(self):
        '''
        In push: push the hashed concatenated dependent data per submodel to the DB (config table?)
        https://stackoverflow.com/questions/7027199/hashing-arrays-in-python
        Here: pull the (max) WAIC from models that are predicting the same dependent data.
        ##We need prior WAIC for this..
        ##Or: we add a column that tells us the input data in the model_config table. Then we can flag models with
        # the same input data and get the max WAIC from comparable models from there.
        #model_score = self.__r2 + (1 - posterior_waic / prior_waic) + (1 - divergences_nb/samples)
        (0.49 + (1 - 6308.74/7000) + (1 - 1299/8000)) / 3 #push_id = '20200609162448'
        (0.49 + (1 - 6308.23 / 7000) + (1 - 1067 / 8000)) / 3  # push_id = '20200609093857'
        '''
        pass

    def __select_split_varnames(self, plot_varnames=None, incl_log=False, incl_interval=False, incl_lowerbound=False, n=6):
        if plot_varnames is None:
            plot_varnames = self.__model.trace.varnames
        selected_varnames = [s for s in self.__model.trace.varnames if any(xs in s for xs in plot_varnames)]
        if incl_log == False:
            selected_varnames = [k for k in selected_varnames if '_log' not in k]
        if incl_interval == False:
            selected_varnames = [k for k in selected_varnames if '_interval' not in k]
        if incl_lowerbound == False:
            selected_varnames = [k for k in selected_varnames if '_lowerbound' not in k]
        plot_vars = [selected_varnames[i:i + n] for i in range(0, len(selected_varnames), n)]
        return plot_vars

    def __get_obs_pred_aggregated_dict(self):
        dict = {}
        for submodel in self.__model.component_container.submodels:
            if 'observed' not in dict:
                dict['observed'] = submodel.dependent.values
            else:
                dict['observed'] = [a + b for a, b in zip(dict['observed'], submodel.dependent.values)]
            if 'predicted' not in dict:
                dict['predicted'] = submodel.result_values
            else:
                dict['predicted'] = [a + b for a, b in zip(dict['predicted'], submodel.result_values)]
        return dict

    # display the total number and percentage of divergent
    def get_divergences(self):
        try:
            divergent = self.__model.trace['diverging']
            divergences = divergent.nonzero()[0].size
            div_perc = divergences / (len(self.__model.trace) * len(self.__model.trace.chains)) * 100
            return (divergences, div_perc)
        except Exception as err:
            print("[!] Could not retrieve divergences due to: {0}".format(err))
            return (-1, -1)

    def get_prior_properties(self, prior_name):
        if ('ela' in prior_name or 'rate' in prior_name) and 'group' not in prior_name:
            run_id = self.__model.run_id
            if 'ela' in prior_name:
                coef = 'ela'
                if 'main' in prior_name:
                    channel = prior_name[prior_name.find('main_') + len('main_'):prior_name.rfind('_ela')]
                else:
                    channel = prior_name[prior_name.find('ia_') + len('ia_'):prior_name.rfind('_in_')]
            else:
                coef = 'rate'
                if 'main' in prior_name:
                    channel = prior_name[prior_name.find('main_') + len('main_'):prior_name.rfind('_rate')]
                else:
                    channel = prior_name[prior_name.find('ia_') + len('ia_'):prior_name.rfind('_in_')]
            puller = pull.Puller(con=config.Config.db_connection_string)
            distribution_type = puller.pull_prior_derived_distribution(run_id, coef)
            mean = puller.pull_group_prior_value(run_id, channel, coef)
            sd = -1.0
        else:
            if prior_name in self.__model.component_container.priors.keys():
                prior = self.__model.component_container.priors[prior_name]
                start = 'pymc3.distributions.continuous.'
                end = "'>"
                raw_type = str(prior.dist)
                distribution_type = raw_type[raw_type.find(start) + len(start):raw_type.rfind(end)]
                a = prior.a
                b = prior.b
                try:
                    if prior.b is None:
                        mean = float(prior.dist.dist(sd=a).mean.eval())
                        sd = float(prior.dist.dist(sd=a).sd.eval())
                    else:
                        mean = float(prior.dist.dist(mu=a, sd=b).mean.eval())
                        sd = float(prior.dist.dist(mu=a, sd=b).sd.eval())
                except:
                    print('Prior mean or sd could not be calculated for distribution type: ' + str(prior.dist))
            else:
                distribution_type = 'prior not found'
                mean = -1
                sd = -1
        return distribution_type, mean, sd

    def print_stat_summary(self):
        with option_context('display.max_rows', None, 'display.max_columns', None):
            print(az.summary(self.__model.trace))

    def show_trace_plot(self, varnames=None, incl_log=False, n=6):
        if varnames is None:
            varnames = self.__model.trace.varnames
        plot_vars = self.__select_split_varnames(plot_varnames=varnames, incl_log=incl_log, n=n)
        for plot_var in plot_vars:
            pm.traceplot(self.__model.trace, varnames=plot_var)
        #plt.show()

    def show_prior_posterior_plot(self, plot_varnames):
        pm.traceplot(self.__model.trace, varnames=plot_varnames, priors=[getattr(self.__model.pm_model[i], 'distribution', None) for i in plot_varnames], combined=True)
        #plt.show()
        plt.savefig('trace_prior_posterior_plot.png')

    def show_prior_posterior_plot_custom(self, plot_varnames=None, incl_log=False, n=8):
        if plot_varnames is None:
            plot_varnames = self.__model.trace.varnames
        plot_vars = self.__select_split_varnames(plot_varnames=plot_varnames, incl_log=incl_log, n=n)
        for plot_vars_page in plot_vars:
            idx_plot = list(range(1, n + 1)) * 100
            for idx, key in enumerate(plot_vars_page):
                if key in self.__model.component_container.priors:
                    a = self.__model.component_container.priors[key].a
                    b = self.__model.component_container.priors[key].b
                    distribution_type, mean, sd = self.get_prior_properties(key)
                    prior_label = distribution_type + ' prior (mean = ' + str(round(mean, 2)) + ', sd = ' + str(round(sd, 2)) + ')'

                    # In half normal, the SD is assigned to a and not to b. reverse that here
                    if distribution_type == 'HalfNormal':
                        b = a
                        a = 0
                        x = np.linspace(0, 3 * b, 100)
                        y = stats.halfnorm.pdf(x, a, b)
                    elif distribution_type == 'Normal':
                        x = np.linspace(a - 3 * b, a + 3 * b, 100)
                        y = stats.norm.pdf(x, a, b)
                    # Lognormal requires adjusted x limits as well as adjusted rounding of mean
                    elif distribution_type == 'Lognormal':
                        if math.exp(a) < 0.01:
                            x_max = 0.01
                            round_decimals = 5
                        else:
                            x_max = math.exp(a) * 4 + 2 * b
                            round_decimals = 2
                        x = np.linspace(0, x_max, 100)
                        y = stats.lognorm.pdf(x, s=sd, scale=mean)
                        prior_label = distribution_type + ' prior (mean = ' + str(round(mean, round_decimals)) + ', sd = ' + str(
                            round(sd, 2)) + ')'
                    elif distribution_type == 'logitnormal':
                        x, y = DistributionProvider.instance().get_pdf(Distribution_Type(distribution_type), a, b)
                    elif distribution_type == 'Bound':
                        x = np.linspace(0, 1, 100)
                        y = stats.norm.pdf(x, a, b)
                    elif distribution_type == 'Beta':
                        x = np.linspace(0, 1, 100)
                        y = stats.beta.pdf(x, a, b)
                    elif distribution_type == 'Gamma':
                        x = np.linspace(0, 10, 100)
                        y = stats.gamma.pdf(x, a, b)
                    else:
                        x = np.linspace(0, 5, 100)
                        y = np.linspace(0, 0, 100)
                        prior_label = 'Prior distribution type not found'
                else:
                    x = np.linspace(0, 5, 100)
                    y = np.linspace(0, 0, 100)
                    prior_label = 'No prior found'
                if idx == 0:
                    plt.figure()
                plt.subplot(2, n / 2, idx_plot[idx])
                color = 'tab:red'
                plt.plot(x, y, label=prior_label, color=color)
                plt.legend(loc=2)
                ax = plt.gca()
                ax.tick_params(axis='y', colors=color)
                plt.twinx()
                color = 'tab:blue'
                posterior = np.array(self.__model.trace.get_values(key))
                if np.mean(posterior) > 0.01:
                    round_mean = 2
                    round_sd = 2
                else:
                    round_mean = 5
                    round_sd = 5
                # Mean
                posterior_label = 'Posterior (mu = ' + str(round(np.mean(posterior), round_mean)) + \
                                  ', sd = ' + str(round(np.std(posterior), round_sd)) + ')'
                plt.xlim(min(x), max(x))
                sn.kdeplot(posterior, label=posterior_label, color=color)
                plt.title(key)
                plt.legend(loc=3)
                ax = plt.gca()
                ax.tick_params(axis='y', colors=color)
            plt.show()

    def show_pred_vs_obs_over_time(self, save_to_xlsx=False, pathname=os.getcwd()):
        i = 0
        observed_type = self.__model.component_container.submodels[0].dependent.name
        if 'lc_orders' in observed_type:
            observed_type_name = 'Orders'
        elif 'lc_revenue' in observed_type:
            observed_type_name = 'Log10(Revenue)'
        else:
            raise (Exception("[!] Observed type not defined for plotting"))
        submodels_dict = Attribution.get_attribution_per_date_and_submodel_and_channel(self.__model.component_container)
        if save_to_xlsx is True:
            writer = pd.ExcelWriter(pathname + '\\' + observed_type_name + '_per_submodel_' + str(self.__push_id) + '.xlsx', engine='xlsxwriter')
            dict_save_submodel = {}
            dict_save_aggregated = {}
        for submodel in self.__model.component_container:
            # chart config
            r2 = round(pm.r2_score(submodel.dependent.values, submodel.result_values)[1], 2)
            fig, axe = plt.subplots(1, 1, figsize=(15, 4))
            axe.set_ylabel(observed_type_name + ' assigned to ' + submodel.main_effect.name.replace('main_', ''))
            axe.set_xlabel('Date')
            plt.title('Submodel: ' + submodel.main_effect.name.replace('main_', '') + ', WAIC = ' + str(
                self.__waic) + ' (full model), R^2 = ' + str(r2) + ' (submodel)')
            axe.plot(submodel.dates, submodel.result_values, label='Predicted', linestyle='dotted')
            axe.plot(submodel.dates, submodel.dependent.values, label='Observed', color='black')
            plot_dict = submodels_dict[submodel.main_effect.channel_name]
            plot_colors = {}
            plot_colors[submodel.main_effect.name] = ColorHash(submodel.main_effect.name.replace('main_', '')).hex
            for ia in submodel.interactions:
                plot_colors[ia.name] = ColorHash(ia.name.replace('ia_', '')).hex
            axe.stackplot(submodel.dates, plot_dict.values(), labels=plot_dict.keys(), colors=plot_colors.values())
            axe.legend(loc=2)
            plt.savefig('preds_vs_obs_with_time_' + str(i) + '.png')
            i = i + 1
            #plt.show()

            if save_to_xlsx is True:
                if 'date' not in dict_save_aggregated.keys():
                    dict_save_aggregated['date'] = [date_obj.strftime('%Y-%m-%d') for date_obj in submodel.dates]
                plot_dict['observed'] = submodel.dependent.values
                plot_dict['predicted'] = submodel.result_values
                for key, value in plot_dict.items():
                    if key not in dict_save_aggregated:
                        dict_save_aggregated[key] = value
                    else:
                        dict_save_aggregated[key] = [a + b for a, b in zip(dict_save_aggregated[key], value)]
                dict_save_submodel['date'] = [date_obj.strftime('%Y-%m-%d') for date_obj in submodel.dates]
                dict_save_submodel.update(plot_dict)
                df = pd.DataFrame.from_dict(dict_save_submodel)
                df.to_excel(writer, sheet_name=submodel.name, index=False)
        if save_to_xlsx is True:
            df = pd.DataFrame.from_dict(dict_save_aggregated)
            df.to_excel(writer, sheet_name='aggregated', index=False)
            writer.save()

    def show_posterior_pred_plot(self):
        ppc = pm.sample_posterior_predictive(self.__model.trace, samples=100, model=self.__model)  # move to regression
        _, ax = plt.subplots(figsize=(12, 6))
        ax.hist([y.mean() for y in ppc['inference']], bins=19, alpha=0.5)
        ax.axvline(self.__model.ic.dependent.values.mean())
        ax.set(title='Posterior predictive of the mean', xlabel='mean(y)', ylabel='Frequency')
        plt.savefig('show_posterior_plot' + '.png')

    def show_autocorrelation_plot(self, plot_varnames=None, incl_log=False, n=6):
        if plot_varnames is None:
            plot_varnames = self.__model.trace.varnames
        plot_vars = self.__select_split_varnames(plot_varnames=plot_varnames, incl_log=incl_log, n=n)
        i = 0
        for plot_var in plot_vars:
            pm.plots.autocorrplot(self.__model.trace, varnames=plot_var, figsize=(17, 5))
            plt.savefig('auto_corr_' + str(i) + '.png')
            i = i + 1
            #plt.show()

    def show_correlogram_plot(self):
        i = 0
        for submodel in self.__model.component_container:
            sn.pairplot(submodel.df, kind="reg")
            plt.savefig('submodel_corr_' + str(i) + '.png')
            i = i + 1
            #plt.show()

    def show_geweke_plot(self, plot_varnames=None, incl_log=False, n=6):
        if plot_varnames is None:
            plot_varnames = self.__model.trace.varnames
        plot_vars = self.__select_split_varnames(plot_varnames=plot_varnames, incl_log=incl_log, n=n)
        score = pm.geweke(self.__model.trace, first=0.1, last=0.5, intervals=20)
        for plot_vars_page in plot_vars:
            idx_plot = list(range(1, n + 1)) * 100
            for idx, key in enumerate(plot_vars_page):
                if idx == 0:
                    plt.figure()
                plt.subplot(2, n / 2, idx_plot[idx])
                for chain in score:
                    plt.scatter(score[chain][key][:, 0], score[chain][key][:, 1], marker='o', s=100, label=chain)
                    plt.axhline(-1.98, c='r')
                    plt.axhline(1.98, c='r')
                    plt.xlabel('First iteration in segment')
                    plt.ylabel('Z-score: first 10% vs last 50% of chain')
                    plt.title(key)
                    plt.ylim(-2.5, 2.5)
                    plt.xlim(0 - 10, .5 * self.__model.trace[key].shape[0] / 2 + 10)
                    plt.title(key)
                    plt.legend(title='chain')
                    plt.show()

    def show_heatmap_plot(self):
        i = 0
        for ic in self.__model.component_container:
            corrMatrix = ic.df.corr()
            plt.figure()
            sn.heatmap(corrMatrix, annot=True)
            plt.savefig('heatmap_plot_' + str(i) + '.png')
            i = i + 1
            #plt.show()

    def show_divergent_transitions_plot(self, plot_varnames=None, incl_log=False, n=16):
        if plot_varnames is None:
            plot_varnames = self.__model.trace.varnames
        plot_vars = self.__select_split_varnames(plot_varnames=plot_varnames, incl_log=incl_log, n=n)
        tracedf = pm.trace_to_dataframe(self.__model.trace)
        tracedf_max = np.max(tracedf, axis=0)
        plot_vars_flat = [item for sublist in plot_vars for item in sublist]
        plot_vars_flat_sorted = [x for _, x in sorted(zip(tracedf_max, plot_vars_flat))]
        tracedf_sorted = tracedf[plot_vars_flat_sorted]
        plot_vars_grouped_sorted = [plot_vars_flat_sorted[i:i + n] for i in range(0, len(plot_vars_flat_sorted), n)]
        divergent = self.__model.trace['diverging']

        for plot_vars_page in plot_vars_grouped_sorted:
            for idx, key in enumerate(plot_vars_page):
                if idx == 0:
                    plt.figure()
                trace_vars_sorted = tracedf_sorted[plot_vars_page]
                trace_min_sorted = np.min(trace_vars_sorted, axis=0)
                trace_max_sorted = np.max(trace_vars_sorted, axis=0)
                plt.plot(trace_min_sorted.values, color='k', alpha=0.5)
                plt.plot(trace_max_sorted.values, color='k', alpha=0.5)
                plt.plot(trace_vars_sorted.values[divergent == 1].T, color='C2', lw=.5)
                plt.ylim([min(trace_min_sorted.values), max(trace_max_sorted.values)])
                plt.xticks(range(len(plot_vars_page)), plot_vars_page, rotation=90)
                plt.tight_layout()
                plt.show()
