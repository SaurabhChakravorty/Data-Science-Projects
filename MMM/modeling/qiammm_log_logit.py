from collections import namedtuple

import arviz as az
import pymc3 as pm
import theano
from numpy import float64 as flt
from theano import tensor as tt
from theano.ifelse import ifelse

from modeling.enums import Control_Type
from modeling.qiammm import QIAmmm

DistParas = namedtuple('DistParas', 'a b')

"""
[i] The pm model is _not_ defined through vector notation on purpose. 
    The element-wise definition is required due to the (dynamic) rate transformation in the fitting.
To dos:
* Is there any other option to speed that up? 
"""


class LogLogitMMM(QIAmmm):

    def __init__(self, run_id, push_id=-1):
        super().__init__(run_id)
        self.__push_id = push_id

    @property
    def push_id(self):
        return self.__push_id

    # [i] Lengthy method as all distributions have to be defined under pm.Model()
    def _fit(self):
        with pm.Model() as model:
            theano.compile.mode.Mode(linker='py', optimizer='fast_compile')
            theano.compile.mode.Mode(linker='cvm', optimizer='fast_run')
            '''
            Create group distributions for each channel. Distributions for both rate and elasticity are created. 
            They are saved in two separate dicts under the channel name.
            '''
            grates = {}
            gelas = {}
            for channel in self.component_container.channel_names:
                for coef in ('rate', 'ela'):
                    # group prior a
                    gprior = self.component_container.group_priors[channel][coef + '_a']
                    gdist_a = gprior.dist(gprior.name, mu=gprior.a, sd=gprior.b)
                    # group prior b
                    gprior = self.component_container.group_priors[channel][coef + '_b']
                    if gprior.b is None:
                        gdist_b = gprior.dist(gprior.name, sd=gprior.a)
                    else:
                        gdist_b = gprior.dist(gprior.name, mu=gprior.a, sd=gprior.b)
                    # add to dict
                    if coef == 'rate':
                        grates[channel] = DistParas(gdist_a, gdist_b)
                    else:
                        gelas[channel] = DistParas(gdist_a, gdist_b)
            '''
            Create control variable distributions. [!] dictionary (sm) of dictionary (control type)
            '''
            sm_controls = {}
            for sm in self.component_container:
                controls = {}
                for control in sm.control_priors.values():
                    if control.b is None:
                        controls[control.name] = control.dist(control.name + '_' + sm.name, sd=control.a)
                    else:
                        controls[control.name] = control.dist(control.name + '_' + sm.name, mu=control.a, sd=control.b)
                sm_controls[sm.name] = controls
            '''
            Create tensor arrays for each submodel. Models are represented by their respective component object.
            '''
            for sm in self.component_container:
                # intercept or 0
                if Control_Type.intercept.value in sm_controls[sm.name]:
                    model_tensor = [sm_controls[sm.name][Control_Type.intercept.value] for _ in self.component_container.range]
                else:
                    model_tensor = [0 for _ in self.component_container.range]
                # trend
                if Control_Type.trend.value in sm_controls[sm.name]:
                    for i, v in enumerate(model_tensor):
                        model_tensor[i] = tt.add(v, tt.mul(sm_controls[sm.name][Control_Type.trend.value], sm.trend_values[i]))
                # weekend dummy
                if Control_Type.we.value in sm_controls[sm.name]:
                    for i, v in enumerate(model_tensor):
                        model_tensor[i] = tt.add(v, tt.mul(sm_controls[sm.name][Control_Type.we.value], sm.we_values[i]))
                # date dummies
                if 'dummy' in sm.control_priors.keys():
                    for d in sm.dummy_values:
                        for i, v in enumerate(model_tensor):
                            model_tensor[i] = tt.add(v, tt.mul(sm_controls[sm.name][d.keys()], d.values()[i]))
                ###
                # main effect
                key = sm.main_effect.channel_name
                rate_dist = self.component_container.derived_priors['rate'].dist(sm.main_effect.rate_name, mu=grates[key].a, sd=grates[key].b)
                ela_dist = self.component_container.derived_priors['ela'].dist(sm.main_effect.ela_name, mu=gelas[key].a, sd=gelas[key].b)
                main_tensor = sm.main_effect.get_sadstock_tensor(rate_dist, ela_dist)
                b_dist = sm.main_effect.prior_b.dist(sm.main_effect.prior_b.name, mu=sm.main_effect.prior_b.a, sd=sm.main_effect.prior_b.b)
                for i, v in enumerate(model_tensor):
                    model_tensor[i] = tt.add(v, tt.mul(b_dist, main_tensor[i]))
                # interactions
                for ia in sm.interactions:
                    key = ia.channel_name
                    rate_dist = self.component_container.derived_priors['rate'].dist(ia.rate_name, mu=grates[key].a, sd=grates[key].b)
                    ela_dist = self.component_container.derived_priors['ela'].dist(ia.ela_name, mu=gelas[key].a, sd=gelas[key].b)
                    ia_tensor = ia.get_sadstock_tensor(rate_dist, ela_dist)
                    b_dist = ia.prior_b.dist(ia.prior_b.name, mu=ia.prior_b.a, sd=ia.prior_b.b)
                    for i, v in enumerate(model_tensor):
                        model_tensor[i] = tt.add(v, tt.mul(b_dist, tt.mul(ia_tensor[i], main_tensor[i])))
                # dummies
                for d in sm.dummies:
                    b_dist = d.prior.dist(d.name, mu=d.prior.a, sd=d.prior.b)
                    for i, v in enumerate(model_tensor):
                        model_tensor[i] = tt.add(v, tt.mul(b_dist, d.values[i]))
                # y sum control
                sum_observations = sum(sm.dependent.values)
                delta_tensor = tt.true_div(tt.sub(tt.sum(model_tensor), flt(sum_observations)), flt(sum_observations))
                delta_dist = pm.Normal.dist(mu=0, sd=0.01)
                pm.Potential('y_sum_' + sm.name, delta_dist.logp(delta_tensor))
                # [!] lognormal inference
                inference_tensor = []
                for i, v in enumerate(model_tensor):
                    inference_tensor.append(ifelse(tt.le(v, flt(1)), flt(0), tt.log(v)))
                sigma = self.component_container.sigma_prior.dist(self.component_container.sigma_prior.name + '_' + sm.name, sd=sm.prior_sd)
                pm.Lognormal('inference_' + sm.name, mu=inference_tensor, sd=sigma, observed=sm.dependent.values)
            # run
            self._pm_model = model
            draws, tune, chains, cores = self._sampling_args
            self._trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores)
            self.set_as_fitted()

    '''
    Public method to re-set posteriors and associated values like sadstock and result_values.
    E.g., after changes in the model
    '''
    def set_as_fitted(self):
        try:
            means = {}
            for metric_key, channels in az.summary(self.trace).items():
                for channel_key, value in channels.items():
                    if metric_key == 'mean':
                        means[channel_key] = value
            for sm in self.component_container:
                sm.set_as_fitted(means)
        except Exception as err:
            print("[!] Could not set regression object to fitted due to: {0}".format(err))
