import gc
from math import inf
from multiprocessing import Process, Value, Manager

import arviz as az
import pymc3 as pm
import theano
from scipy.optimize import fmin_cobyla

from db_interaction.entity_gateway import EntityGateway
from modeling.enums import Observed_Type
from modeling.qiammm import QIAmmm
from fitting.platform_manager import PlatformManager
from prior_factory.log_logit import Attribution_Type
from prior_factory.log_logit import LogLogit
from prior_factory.prior_predictive_sampler import PPS_LogLogit

"""
Notes:
* [!] Includes # y sum control for testing
* [i] There are two different implementations for 'fit_pm_model'; keep that in mind if sth. is changed in there
* maxfun = nb of channels * 2 (ela + rate) * 2.5 => 15 channels = 38 opt steps
* Uses PPS_LogLogit to sample prior dependent SDs with very small logit-normal SDs (as this is a very good proxy)
* Uses flag: __dependent_sds_run, to only estimate the prior dependent sds once. This can be changed if estimating them more often turns out to be beneficial.
* y-sum control:
    from theano import tensor as tt
    from numpy import float64 as flt
    sum_observations = sum(sm.dependent.values)
    delta_tensor = tt.true_div(tt.sub(tt.sum(model_tensor), flt(sum_observations)), flt(sum_observations))
    delta_dist = pm.Normal.dist(mu=0, sd=0.001)
    pm.Potential('y_sum_' + sm.name, delta_dist.logp(delta_tensor))

To dos:
* [!] Does interact with general prior table in the DB --> use specific one
"""

# region module_level_methods
'''
[i] Methods have to be defined outside of the class on module level to allow them to be pickled
'''

def fit_pm_model(waic, elas, rates, cc, sampling_args):
    try:
        with pm.Model() as model:
            theano.compile.mode.Mode(linker='py', optimizer='fast_compile')
            for sm in cc:
                # main effect
                b_dist = sm.main_effect.prior_b.dist(sm.main_effect.prior_b.name, mu=sm.main_effect.prior_b.a, sd=sm.main_effect.prior_b.b)
                sadstocks = sm.main_effect.get_sadstocks(rates[sm.main_effect.rate_name], elas[sm.main_effect.ela_name])
                model_tensor = b_dist * sadstocks
                # interactions
                for ia in sm.interactions:
                    b_dist = ia.prior_b.dist(ia.prior_b.name, mu=ia.prior_b.a, sd=ia.prior_b.b)
                    sadstocks_ia = ia.get_sadstocks(rates[ia.rate_name], elas[ia.ela_name])
                    model_tensor += b_dist * sadstocks_ia * sadstocks
                for d in sm.dummies:
                    b_dist = d.prior.dist(d.name, mu=d.prior.a, sd=d.prior.b)
                    model_tensor += b_dist * d.values
                # build sm ----> whats getting build here? : The Model itself to sample from distributions randomly
                sigma = cc.sigma_prior.dist(cc.sigma_prior.name + '_' + sm.name, sd=sm.prior_sd)
                pm.Lognormal('inference_' + sm.name, mu=pm.math.log(model_tensor), sd=sigma, observed=sm.dependent.values)
            # run
            draws, tune, chains, cores = sampling_args
            trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores)
            waic.value = round(pm.waic(trace, model).WAIC, 2)
    except Exception as e:
        with open('fitting_errors.txt', 'a') as f:
            f.write('failed due to: ' + str(e) + '\n')

def set_dependent_sds(shared_dict, run_id, start_and_end_date):
    sds = PPS_LogLogit.instance().get_dependent_sds(
        run_id=run_id,
        dates=start_and_end_date,
        nb_of_samples=PlatformManager.instance().prior_samples,
        observed_type=Observed_Type.lc_orders)
    for key in sds.keys():
        shared_dict[key] = sds[key]

# endregion

class LogConstMMM(QIAmmm):

    def __init__(self, run_id, push_id, model_name):
        super().__init__(run_id)
        self.__push_id = push_id
        self.__model_name = model_name
        self.__opt_step = 0
        self.__fit_all_coefs = False
        self.__ini_value = 0.75 # 0.75
        self.__channel_indexes = {}
        self.__waic = inf
        self.__elas = None
        self.__rates = None
        self.__dependent_sds_run = False
        # [i] temporary variables used in optimization:
        self.__elas_tmp = None
        self.__rates_tmp = None
        self.__pm_model_tmp = None
        self.__trace_tmp = None

    @property
    def push_id(self):
        return self.__push_id

    @property
    def fit_all_coefs(self):
        return self.__fit_all_coefs

    @fit_all_coefs.setter
    def fit_all_coefs(self, fit_all_coefs):
        self.__fit_all_coefs = fit_all_coefs

    def _fit(self):
        if self.__fit_all_coefs:
            self.__fit_all_coefficients()
        else:
            self.__run_conversion_coefficients_only()

    def __run_conversion_coefficients_only(self):
        self.__set_elas_and_rates_from_db()
        self.__set_priors()
        self.__set_dependent_variable_sds()
        self.__fit_pm_model()
        self.set_as_fitted()

    def __fit_all_coefficients(self):
        self.__set_channel_indexes()
        # x = [self.__ini_value for _ in self.__channel_indexes.keys()]
        elas, rates = self.__get_elas_and_rates_from_db()
        elaras = {**elas, **rates}
        elaras = {k: 0.75 if v > 0.75 else v for k, v in elaras.items()}
        x = [elaras[key] for key in self.__channel_indexes.keys()]
        self.__opt_step = 0
        maxfun = int(len(self.component_container.get_channel_names()) * 4)
        result = fmin_cobyla(func=self.__minimize_waic, x0=x, cons=[], rhobeg=0.2, rhoend=0.0005, maxfun=maxfun)
        print('[i] COBYLA finished')
        self.__apply_ela_and_rate_bounds(result)
        #print(result)
        self.__try_push_step_results(result, self.__opt_step, True, True, self.__waic)
        print('final run...')
        self.__fit_pm_model()
        self.set_as_fitted()
        self.__elas_tmp = None
        self.__rates_tmp = None
        self.__pm_model_tmp = None
        self.__trace_tmp = None

    def __set_elas_and_rates_from_db(self):
        _elas, _rates = self.__get_elas_and_rates_from_db()
        # [i] transfer to dictionaries aligned with model structure
        elas, rates = {}, {}
        for sm in self.component_container:
            elas[sm.main_effect.ela_name] = _elas[(sm.main_effect.channel_name, 'ela')]
            rates[sm.main_effect.rate_name] = _rates[(sm.main_effect.channel_name, 'rate')]
            for ia in sm.interactions:
                elas[ia.ela_name] = _elas[(ia.channel_name, 'ela')]
                rates[ia.rate_name] = _rates[(ia.channel_name, 'rate')]
        # [i] set corresponding attributes
        self.__elas, self.__rates = elas, rates

    def __get_elas_and_rates_from_db(self):
        l = EntityGateway.get_priors_group_values(self._run_id)
        # [i] transfer to dictionaries for easy access
        elas, rates = {}, {}
        for v in l:
            if v[1] == 'rate':
                rates[(v[0], v[1])] = v[2]
            elif v[1] == 'ela':
                elas[(v[0], v[1])] = v[2]
            else:
                raise Exception("[!] Wrong input from DB: ela & rate values")
        return elas, rates

    def __set_channel_indexes(self):
        elas = {}
        j = 0
        for i, channel in enumerate(self.component_container.get_channel_names()):
            elas[(channel, 'ela')] = i
            j = i
        rates = {}
        for channel in self.component_container.get_channel_names():
            j = j + 1
            rates[(channel, 'rate')] = j
        self.__channel_indexes = {**elas, **rates}

    def __minimize_waic(self, x):
        self.__apply_ela_and_rate_bounds(x)
        print('[i] Initializing new run')
        self.__initialize(x)
        # [i] to speed up initialization, a new process is created
        print('[i] PyMC3 model called')
        waic_tmp = Value('f', inf)
        p = Process(target=fit_pm_model, args=(waic_tmp, self.__elas_tmp, self.__rates_tmp, self.component_container, self._sampling_args))
        p.start()
        p.join()
        self.__opt_step += 1
        is_improvement = self.__waic == inf or self.__waic > waic_tmp.value
        if is_improvement:
            self.__waic = waic_tmp.value
            self.__elas = self.__elas_tmp
            self.__rates = self.__rates_tmp
            self._pm_model = self.__pm_model_tmp
            self._trace = self.__trace_tmp
        self.__try_push_step_results(x, self.__opt_step, is_improvement, False, waic_tmp.value)
        # [i] release memory before doing another run
        gc.collect()
        print('[i] Step result (waic): ' + str(waic_tmp.value))
        return waic_tmp.value

    def __apply_ela_and_rate_bounds(self, x):
        for i, v in enumerate(x):
            if v >= 1:
                x[i] = 0.99 # try also 0.95
            elif v <= 0:
                x[i] = 0.01 # try also 0.05

    def __initialize(self, x):
        self.__transfer_group_value_vector(x)
        self.__set_priors(x)
        if not self.__dependent_sds_run:
            self.__set_dependent_variable_sds()
            self.__dependent_sds_run = True

    def __transfer_group_value_vector(self, x):
        elas, rates = {}, {}
        for sm in self.component_container:
            elas[sm.main_effect.ela_name] = x[self.__channel_indexes[(sm.main_effect.channel_name, 'ela')]]
            rates[sm.main_effect.rate_name] = x[self.__channel_indexes[(sm.main_effect.channel_name, 'rate')]]
            for ia in sm.interactions:
                elas[ia.ela_name] = x[self.__channel_indexes[(ia.channel_name, 'ela')]]
                rates[ia.rate_name] = x[self.__channel_indexes[(ia.channel_name, 'rate')]]
        self.__elas_tmp, self.__rates_tmp = elas, rates

    def __try_push_step_results(self, x, step, is_improvement, is_final, waic):
        try:
            d = []
            if waic == inf:
                waic = -1
            for i, channel_name in enumerate(self.component_container.get_channel_names()):
                d.append((self.__push_id, self.__model_name, step, channel_name, 'ela', x[self.__channel_indexes[(channel_name, 'ela')]], is_improvement, is_final, waic))
                d.append((self.__push_id, self.__model_name, step, channel_name, 'rate', x[self.__channel_indexes[(channel_name, 'rate')]], is_improvement, is_final, waic))
            EntityGateway.push_opt_steps(d)
            print('[i] Pushed step results to DB')
        except Exception as e:
            print('[!] Could not push step results due to:', str(e))

    def __set_priors(self, x=None):
        priors_group_values = None
        if x is not None:
            priors_group_values = [(key[0], key[1], x[self.__channel_indexes[key]]) for key in self.__channel_indexes.keys()]
        bp = LogLogit(
            run_id=self._run_id,
            sd=0.001,  # [i] very small; dependency with LogLogitPriorPredictiveSampler
            dates=self.component_container.submodels[0].start_and_end_date,
            attribution_type=Attribution_Type.tor,
            priors_group_values=priors_group_values)
        bp.run()
        self._set_prior_bs()

    def __set_dependent_variable_sds(self):
        print('[i] Sampling dependent sds...')
        manager = Manager()
        d = manager.dict()
        for sm in self.component_container:
            d[sm.name] = 0.0
        p = Process(target=set_dependent_sds, args=(d, self._run_id, self.component_container.submodels[0].start_and_end_date))
        p.start()
        p.join()
        for sm in self.component_container:
            sm.prior_sd = float(d[sm.name])

    def __fit_pm_model(self):
        theano.compile.mode.Mode(linker='cvm', optimizer='fast_run')
        with pm.Model() as model:
            for sm in self.component_container:
                # main effect
                b_dist = sm.main_effect.prior_b.dist(sm.main_effect.prior_b.name, mu=sm.main_effect.prior_b.a, sd=sm.main_effect.prior_b.b)
                sadstocks = sm.main_effect.get_sadstocks(self.__rates[sm.main_effect.rate_name], self.__elas[sm.main_effect.ela_name])
                model_tensor = b_dist * sadstocks
                # interactions
                for ia in sm.interactions:
                    b_dist = ia.prior_b.dist(ia.prior_b.name, mu=ia.prior_b.a, sd=ia.prior_b.b)
                    sadstocks_ia = ia.get_sadstocks(self.__rates[ia.rate_name], self.__elas[ia.ela_name])
                    model_tensor += b_dist * sadstocks_ia * sadstocks
                for d in sm.dummies:
                    b_dist = d.prior.dist(d.name, mu=d.prior.a, sd=d.prior.b)
                    model_tensor += b_dist * d.values
                sigma = self.component_container.sigma_prior.dist(self.component_container.sigma_prior.name + '_' + sm.name, sd=sm.prior_sd)
                pm.Lognormal('inference_' + sm.name, mu=pm.math.log(model_tensor), sd=sigma, observed=sm.dependent.values)
            # run
            self._pm_model = model
            draws, tune, chains, cores = self._sampling_args
            self._trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores)

    def __constr_within_one(self, x):
        for v in x:
            if v <= 0:
                print('[i] Constr. violation: v <= 0')
                return v - 0.01
            if v >= 1:
                print('[i] Constr. violation: v >= 1')
                return 1 - v - 0.01

    def set_as_fitted(self):
        try:
            means = {**self.__elas, **self.__rates}
            for metric_key, channels in az.summary(self._trace).items():
                for channel_key, value in channels.items():
                    if metric_key == 'mean':
                        means[channel_key] = value
            for sm in self.component_container:
                sm.set_as_fitted(means)
        except Exception as err:
            print("[!] Could not set regression object to fitted due to: {0}".format(err))
