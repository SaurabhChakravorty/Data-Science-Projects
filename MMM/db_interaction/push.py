import bz2
import datetime
import math
import pickle
import sys

import arviz as az

import config
import db_interaction.db_helper as helper
from db_interaction import pull
from evaluation.attribution import Attribution
from evaluation.result_reporter import ResultReporter
from db_interaction.entity_gateway import EntityGateway

"""
To dos:
* [!] Use try-catch more consistently. Use it to catch DB errors only. Catch other errors explicitly.
* Some push methods have parameters some don't: make it more consistent or explain why this is the case
"""


class Pusher():

    def __init__(self, model, run_id, push_id, dates, include_we, model_score=-1):
        self.__model = model
        self.__run_id = run_id
        self.__push_id = push_id
        self.__dates = dates
        self.__include_we = include_we
        self.__model_score = model_score

    @property
    def push_id(self):
        return self.__push_id

    @classmethod
    def get_new_push_id(cls):
        return helper.dt_to_integer(datetime.datetime.now())

    def push_model(self):
        cnx = helper.get_cnx()
        cursor = cnx.cursor()
        try:
            model = pickle.dumps(self.__model)
            compr_model = bz2.compress(model)
            q_push = ("insert into mmm.res_model_objects (push_id, model_name, waic, object) values (?, ?, ?, ?)")
            data = (self.__push_id, self.__model.name, str(self.__model_score), compr_model)
            cursor.execute(q_push, data)
            cnx.commit()
            print('[i] pushed model')
        except:
            q_push = ("insert into mmm.res_errors (push_id, error_type, message) values (?, ?, ?)")
            data = (self.__push_id, 'push model object ' + helper.get_rnd_chars(3), str(sys.exc_info()))
            cursor.execute(q_push, data)
            cnx.commit()
            print('[!] error while pushing model object')
        cursor.close()
        cnx.close()

    def push_sampling_results(self):
        cnx = helper.get_cnx()
        cursor = cnx.cursor()
        add = ("insert into mmm.res_sampling values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
        # dict for each metric
        means = {}
        sds = {}
        hpd_lowers = {}
        hpd_uppers = {}
        mcse_mean = {}
        mcse_sd = {}
        ess_mean = {}
        ess_sd = {}
        ess_bulk = {}
        ess_tail = {}
        rhats = {}
        # transferring data to each dict
        for metric_key, channels in az.summary(self.__model.trace).items():
            for ch_key, value in channels.items():
                if math.isnan(value):
                    value = 'NULL'
                if metric_key == 'mean':
                    means[ch_key] = value
                elif metric_key == 'sd':
                    sds[ch_key] = value
                elif metric_key == 'hpd_3%':
                    hpd_lowers[ch_key] = value
                elif metric_key == 'hpd_97%':
                    hpd_uppers[ch_key] = value
                elif metric_key == 'mcse_mean':
                    mcse_mean[ch_key] = value
                elif metric_key == 'mcse_sd':
                    mcse_sd[ch_key] = value
                elif metric_key == 'ess_mean':
                    ess_mean[ch_key] = value
                elif metric_key == 'ess_sd':
                    ess_sd[ch_key] = value
                elif metric_key == 'ess_bulk':
                    ess_bulk[ch_key] = value
                elif metric_key == 'ess_tail':
                    ess_tail[ch_key] = value
                elif metric_key == 'r_hat':
                    rhats[ch_key] = value
        # pushing to db, key = channel name
        for key in means:
            data = (
                self.__push_id,
                str(self.__model_score),
                self.__model.name,
                key,
                means[key],
                sds[key],
                hpd_lowers[key],
                hpd_uppers[key],
                mcse_mean[key],
                mcse_sd[key],
                ess_mean[key],
                ess_sd[key],
                ess_bulk[key],
                ess_tail[key],
                rhats[key]
            )
            try:
                cursor.execute(add, data)
                cnx.commit()
            except:
                q_push = ("insert into mmm.res_errors (push_id, error_type, message) values (?, ?, ?)")
                data = (self.__push_id, 'push model object ' + helper.get_rnd_chars(3), str(sys.exc_info()))
                cursor.execute(q_push, data)
                cnx.commit()
                print('[!] error while pushing samples')
        print('[i] pushed sampling results')
        cursor.close()
        cnx.close()

    def push_diagnostics(self, waic=-1, r2_regular_train=-1, r2_regular_test=-1, r2_bayesian_train=-1, r2_bayesian_test=-1,
                         r_hat_max=-1, divergences_nb=-1, divergences_perc=-1, run_time=-1):
        cnx = helper.get_cnx()
        cursor = cnx.cursor()
        try:
            q_push = ("insert into mmm.res_diagnostics (push_id, model_name, model_score, waic, r2_regular_train, "
                      "r2_regular_test, r2_bayesian_train, r2_bayesian_test, r_hat_max, divergences_nb, "
                      "divergences_perc, run_time) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
            data = (self.__push_id, self.__model.name, str(self.__model_score), waic, r2_regular_train, r2_regular_test,
                    r2_bayesian_train, r2_bayesian_test, r_hat_max, divergences_nb, divergences_perc, str(run_time))
            cursor.execute(q_push, data)
            cnx.commit()
            print('[i] pushed diagnostics')
        except:
            q_push = ("insert into mmm.res_errors (push_id, error_type, message) values (?, ?, ?)")
            data = (self.__push_id, 'push diagnostics ' + helper.get_rnd_chars(3), str(sys.exc_info()))
            cursor.execute(q_push, data)
            cnx.commit()
            print('[!] error while pushing diagnostics')
        cursor.close()
        cnx.close()

    def push_aggregated_orders(self):
        # dict for each metric
        channel_name = {}
        observed = {}
        fitted = {}
        delta_perc_fit_obs = {}
        attributed = Attribution.get_aggregated_orders_per_channel(self.__model.component_container)
        actions = {}
        fitted_per_action = {}
        attributed_per_action = {}

        # Looping can only be done over submodels, because otherwise we can't access the observed and fitted values
        for submodel in self.__model.component_container:
            main_name = submodel.main_effect.name.replace('main_', '')
            channel_name[main_name] = main_name
            observed[main_name] = sum(submodel.dependent.values)
            fitted[main_name] = sum(submodel.result_values)
            delta_perc_fit_obs[main_name] = (fitted[main_name] - observed[main_name]) / observed[main_name] * 100
            # Loop over the interactions. If the interaction does not have a submodel, create its key as well
            for ia in submodel.interactions:
                ia_name = ia.name.replace('ia_', '')
                if ia_name not in observed.keys():
                    observed[ia_name] = None
                    fitted[ia_name] = None
                    channel_name[ia_name] = ia_name
                    delta_perc_fit_obs[ia_name] = None

        # Add actions and conversion rates
        components = EntityGateway.get_distinct_list_of_model_components(self.__run_id)
        for cpn in components:
            actions[cpn] = sum(EntityGateway.get_channel_values(self.__run_id, cpn, 'value',
                                                                self.__dates[0], self.__dates[1], self.__include_we))
            if fitted[cpn] is not None:
                fitted_per_action[cpn] = fitted[cpn] / actions[cpn]
            else:
                fitted_per_action[cpn] = None
            attributed_per_action[cpn] = attributed[cpn] / actions[cpn]

        # New for inserting results into the DB, because not all channels have a submodel. Get data ready and insert:
        cnx = helper.get_cnx()
        cursor = cnx.cursor()
        add = ("insert into mmm.res_aggregated_orders values (?, ?, ?, ?, ?, ?, ?, ?, ?)")

        for key in attributed:
            data = (self.__push_id, channel_name[key], observed[key], fitted[key], delta_perc_fit_obs[key],
                    attributed[key], actions[key], fitted_per_action[key], attributed_per_action[key])
            try:
                cursor.execute(add, data)
                cnx.commit()
            except:
                q_push = ("insert into mmm.res_errors (push_id, error_type, message) values (?, ?, ?)")
                data = [self.__push_id, 'push model object ' + helper.get_rnd_chars(3), str(sys.exc_info())]
                cursor.execute(q_push, data)
                cnx.commit()
                print('[!] error while pushing aggregated orders')

        # Get data of the summed row of all submodels ready and insert
        observed_sum = sum(filter(None, observed.values()))
        fitted_sum = sum(filter(None, fitted.values()))
        delta_perc_fit_obs_sum = (fitted_sum - observed_sum) / observed_sum * 100
        attributed_sum = sum(filter(None, attributed.values()))
        actions_sum = sum(actions.values())
        fitted_per_action_sum = fitted_sum / actions_sum
        attributed_per_action_sum = attributed_sum / actions_sum

        data_sum = (self.__push_id, 'sum', observed_sum, fitted_sum, delta_perc_fit_obs_sum, attributed_sum,
                    actions_sum, fitted_per_action_sum, attributed_per_action_sum)
        try:
            cursor.execute(add, data_sum)
            cnx.commit()
            print('[i] pushed aggregated orders')
        except:
            q_push = ("insert into mmm.res_errors (push_id, error_type, message) values (?, ?, ?)")
            data = [self.__push_id, 'push aggregated orders ' + helper.get_rnd_chars(3), str(sys.exc_info())]
            cursor.execute(q_push, data)
            cnx.commit()
            print('[!] error while pushing aggregated orders')
        cursor.close()
        cnx.close()

    def push_prior_posterior_mean_sd(self, push_id):
        puller = pull.Puller(con=config.Config.db_connection_string)
        posteriors_df = puller.pull_posterior_mean_sd(push_id)
        posteriors_df = posteriors_df.set_index(posteriors_df.loc[:, 'coefficient'])
        reporter = ResultReporter(self.__model, [push_id])

        cnx = helper.get_cnx()
        cursor = cnx.cursor()
        add = ("insert into mmm.res_priors_posteriors values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
        for key in posteriors_df.index:
            posterior_mean = round(posteriors_df.loc[key, 'mean'], 3)
            posterior_sd = round(posteriors_df.loc[key, 'sd'], 3)
            try:
                distribution_type, prior_mean, prior_sd = reporter.get_prior_properties(key)
                prior_mean = round(prior_mean, 3)
                prior_sd = round(prior_sd, 3)
                if prior_mean != 0:
                    perc_diff_mean = round((posterior_mean - prior_mean) / prior_mean * 100, 3)
                else:
                    perc_diff_mean = -1.0
                if prior_mean != 0:
                    perc_diff_sd = round((posterior_sd - prior_sd) / prior_sd * 100, 3)
                else:
                    perc_diff_sd = -1.0
                data = (int(posteriors_df.loc[key, 'push_id']), posteriors_df.loc[key, 'model_name'],
                        posteriors_df.loc[key, 'coefficient'], distribution_type,
                        prior_mean, posterior_mean, perc_diff_mean,
                        prior_sd, posterior_sd, perc_diff_sd)
                cursor.execute(add, data)
                cnx.commit()
            except:
                q_push = ("insert into mmm.res_errors (push_id, error_type, message) values (?, ?, ?)")
                data = [push_id, 'push priors posteriors ' + helper.get_rnd_chars(3), str(sys.exc_info())]
                cursor.execute(q_push, data)
                cnx.commit()
                print('[!] error while pushing priors vs posteriors')
        print('[i] pushed priors vs posteriors')
        cursor.close()
        cnx.close()

    def push_model_configuration(self, waic=-1, r2_regular_train=-1, r2_regular_test=-1, r2_bayesian_train=-1, r2_bayesian_test=-1):
        cnx = helper.get_cnx()
        cursor = cnx.cursor()
        try:
            q_push = ("insert into mmm.res_model_config values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)")
            data = (self.__push_id, self.__model.name, 'NULL', self.__model.component_container.submodels[0].dates[0].year,
                    'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL',
                    waic, r2_regular_train, r2_regular_test, r2_bayesian_train, r2_bayesian_test, str(self.__model_score), 'NULL')
            cursor.execute(q_push, data)
            cnx.commit()
            print('[i] pushed model config')
        except:
            q_push = ("insert into mmm.res_errors (push_id, error_type, message) values (?, ?, ?)")
            data = (self.__push_id, 'push config table entry' + helper.get_rnd_chars(3), str(sys.exc_info()))
            cursor.execute(q_push, data)
            cnx.commit()
            print('[!] error while pushing config table entry')
        cursor.close()
        cnx.close()
