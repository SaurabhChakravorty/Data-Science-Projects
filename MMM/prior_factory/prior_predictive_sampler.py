from collections import defaultdict

from numpy import log, std, pi, median

from db_interaction.entity_gateway import EntityGateway
from modeling.qiammm_log_logit import LogLogitMMM
from modeling.submodel import Submodel
from singleton_pattern import Singleton

'''
Notes:
* PPS = Prior Predictive Sampler

To dos:
* Include control variables
* Push results to DB
'''


@Singleton
class PPS_LogLogit:

    # region prior_model

    def get_prior_model(self, run_id, sd, dates, attribution, observed_type, include_we=False):
        # calculate priors
        from prior_factory.log_logit import LogLogit
        bp = LogLogit(run_id=run_id, sd=sd, dates=dates, attribution_type=attribution, observed_type=observed_type)
        bp.run()
        # initialize model
        from modeling.qiammm_log_const import LogConstMMM
        model = LogConstMMM(run_id, None, None)
        submodel_names = EntityGateway.get_distinct_list_of_submodels(run_id)
        for name in submodel_names:
            model.add_submodel(Submodel(run_id, name, dates, include_we, observed_type=observed_type))
        model.close()
        return self.__get_prior_model(model)

    def __get_prior_model(self, model):
        values = {}
        # b values
        for sm in model.component_container:
            values[sm.main_effect.prior_b.name] = sm.main_effect.prior_b.dist.dist(mu=sm.main_effect.prior_b.a, sd=sm.main_effect.prior_b.b).mean.eval()
            for ia in sm.interactions:
                values[ia.prior_b.name] = ia.prior_b.dist.dist(mu=ia.prior_b.a, sd=ia.prior_b.b).mean.eval()
        # group distribution values
        for channel in model.component_container.channel_names:
            for coef in ('rate', 'ela'):
                # group prior a
                gprior = model.component_container.group_priors[channel][coef + '_a']
                values[gprior.name] = gprior.dist.dist(mu=gprior.a, sd=gprior.b).mean.eval()
                # group prior b
                gprior = model.component_container.group_priors[channel][coef + '_b']
                if gprior.b is None:
                    values[gprior.name] = gprior.dist.dist(sd=gprior.a).mean.eval()
                else:
                    values[gprior.name] = gprior.dist.dist(mu=gprior.a, sd=gprior.b).mean.eval()
        # concrete ela and rate values
        for sm in model.component_container:
            # main ela
            mu, sd = (values[sm.main_effect.channel_name + '_group_ela_a'], values[sm.main_effect.channel_name + '_group_ela_b'])
            values[sm.main_effect.ela_name] = model.component_container.derived_priors['ela'].dist.dist(mu=mu, sd=sd).median.eval()
            # main rate
            mu, sd = (values[sm.main_effect.channel_name + '_group_rate_a'], values[sm.main_effect.channel_name + '_group_rate_b'])
            values[sm.main_effect.rate_name] = model.component_container.derived_priors['rate'].dist.dist(mu=mu, sd=sd).median.eval()
            # ias
            for ia in sm.interactions:
                # ela
                mu, sd = (values[ia.channel_name + '_group_ela_a'], values[ia.channel_name + '_group_ela_b'])
                values[ia.ela_name] = model.component_container.derived_priors['ela'].dist.dist(mu=mu, sd=sd).median.eval()
                # rate
                mu, sd = (values[ia.channel_name + '_group_rate_a'], values[ia.channel_name + '_group_rate_b'])
                values[ia.rate_name] = model.component_container.derived_priors['rate'].dist.dist(mu=mu, sd=sd).median.eval()
        # generate results
        for sm in model.component_container:
            sm.set_as_fitted(values)
        return model

    # endregion

    # region dependent_sds

    def get_dependent_sds(self, run_id, dates, nb_of_samples, observed_type, include_we=False):
        # initialize model
        model = LogLogitMMM(run_id)
        submodel_names = EntityGateway.get_distinct_list_of_submodels(run_id)
        for name in submodel_names:
            model.add_submodel(Submodel(run_id, name, dates, include_we, observed_type=observed_type))
        model.close()
        medians = self.__get_dependent_medians(model, nb_of_samples)
        # get lognormal sds and transform to half-normal sd
        sds = {}
        for sm in model.component_container:
            sd = std([log(v) for v in medians[sm.name]])
            halfnormal_sd = float((sd * pi ** 0.5) / (2 ** 0.5))
            sds[sm.name] = halfnormal_sd
        # fyi print
        for model, sd in sds.items():
            print('[i] ' + str(model) + ': ' + str(sd))
        return sds

    def __get_dependent_medians(self, model, nb_of_samples):
        dists = self.__get_distributions(model)
        samples = self.__get_samples(model, dists, nb_of_samples)
        output = defaultdict(list)
        for i in range(0, nb_of_samples):
            dict = {}
            for key, values in samples.items():
                dict[key] = values[i]
            for sm in model.component_container:
                sm.set_as_fitted(dict)
                output[sm.name].append(median(sm.result_values))
        return output

    def __get_distributions(self, model):
        # set all distributions except concrete ela and rate distributions (these are derived)
        distributions = {}
        # group dists
        for channel in model.component_container.channel_names:
            for coef in ('rate', 'ela'):
                # group dist a
                gprior = model.component_container.group_priors[channel][coef + '_a']
                distributions[gprior.name] = gprior.dist.dist(mu=gprior.a, sd=gprior.b)
                # group dist b
                gprior = model.component_container.group_priors[channel][coef + '_b']
                if gprior.b is None:
                    distributions[gprior.name] = gprior.dist.dist(sd=gprior.a)
                else:
                    distributions[gprior.name] = gprior.dist.dist(mu=gprior.a, sd=gprior.b)
        # b dists
        for sm in model.component_container:
            distributions[sm.main_effect.prior_b.name] = sm.main_effect.prior_b.dist.dist(mu=sm.main_effect.prior_b.a, sd=sm.main_effect.prior_b.b)
            for ia in sm.interactions:
                distributions[ia.prior_b.name] = ia.prior_b.dist.dist(mu=ia.prior_b.a, sd=ia.prior_b.b)
        return distributions

    def __get_samples(self, model, dists, nb_of_samples):
        samples = defaultdict(list)
        # b values
        for key, coef in dists.items():
            samples[key] = coef.random(size=nb_of_samples)
        # group ela and rate values
        for sm in model.component_container:
            # main ela
            samples[sm.main_effect.channel_name + '_group_ela_a'] = dists[sm.main_effect.channel_name + '_group_ela_a'].random(size=nb_of_samples)
            samples[sm.main_effect.channel_name + '_group_ela_b'] = dists[sm.main_effect.channel_name + '_group_ela_b'].random(size=nb_of_samples)
            # main rate
            samples[sm.main_effect.channel_name + '_group_rate_a'] = dists[sm.main_effect.channel_name + '_group_rate_a'].random(size=nb_of_samples)
            samples[sm.main_effect.channel_name + '_group_rate_b'] = dists[sm.main_effect.channel_name + '_group_rate_b'].random(size=nb_of_samples)
            # ias
            for ia in sm.interactions:
                # ela
                samples[ia.channel_name + '_group_ela_a'] = dists[ia.channel_name + '_group_ela_a'].random(size=nb_of_samples)
                samples[ia.channel_name + '_group_ela_b'] = dists[ia.channel_name + '_group_ela_b'].random(size=nb_of_samples)
                # rate
                samples[ia.channel_name + '_group_rate_a'] = dists[ia.channel_name + '_group_rate_a'].random(size=nb_of_samples)
                samples[ia.channel_name + '_group_rate_b'] = dists[ia.channel_name + '_group_rate_b'].random(size=nb_of_samples)
        # concrete ela & rate
        for i in range(0, nb_of_samples):
            for sm in model.component_container:
                # main ela
                mu, sd = (samples[sm.main_effect.channel_name + '_group_ela_a'][i], samples[sm.main_effect.channel_name + '_group_ela_b'][i])
                samples[sm.main_effect.ela_name].append(model.component_container.derived_priors['ela'].dist.dist(mu=mu, sd=sd).random())
                # main rate
                mu, sd = (samples[sm.main_effect.channel_name + '_group_rate_a'][i], samples[sm.main_effect.channel_name + '_group_rate_b'][i])
                samples[sm.main_effect.rate_name].append(model.component_container.derived_priors['rate'].dist.dist(mu=mu, sd=sd).random())
                # ias
                for ia in sm.interactions:
                    # ela
                    mu, sd = (samples[ia.channel_name + '_group_ela_a'][i], samples[ia.channel_name + '_group_ela_b'][i])
                    samples[ia.ela_name].append(model.component_container.derived_priors['ela'].dist.dist(mu=mu, sd=sd).random())
                    # rate
                    mu, sd = (samples[ia.channel_name + '_group_rate_a'][i], samples[ia.channel_name + '_group_rate_b'][i])
                    samples[ia.rate_name].append(model.component_container.derived_priors['rate'].dist.dist(mu=mu, sd=sd).random())
        return samples

    # endregion
