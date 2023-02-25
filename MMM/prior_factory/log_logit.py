from enum import Enum

from numpy import mean, log, abs, sqrt

from db_interaction.entity_gateway import EntityGateway
from modeling.interaction import Interaction
from modeling.main_effect import MainEffect
from modeling.observed import Observed
from modeling.enums import Observed_Type

"""
To dos:
* include_we as parameter
* Include parameter checks
* Push associated priors, like derived priors
"""


class Attribution_Type(Enum):
    tor = 'tor'
    last_touch = 'last_touch'


class LogLogit:

    def __init__(self, run_id, sd, dates, attribution_type, priors_group_values=None, observed_type=Observed_Type.lc_orders):
        if not isinstance(attribution_type, Attribution_Type):
            raise TypeError("[!] 'attribution_type' of LogLogit must be of type Attribution_Type [enum]")
        self.__run_id = run_id
        self.__sd = float(sd)
        self.__dates = dates
        self.__attribution_type = attribution_type
        self.__priors_group_values = {}
        if priors_group_values is None:
            _priors_group_values = EntityGateway.get_priors_group_values(run_id)
        else:
            _priors_group_values = priors_group_values
        for value in _priors_group_values:
            self.__priors_group_values[(value[0], value[1])] = value[2]
        self.__observed_type = observed_type

    @property
    def attribution_type(self):
        return self.__attribution_type

    @attribution_type.setter
    def attribution_type(self, value):
        self.__attribution_type = value

    def run(self):
        self.__push_group_priors()
        self.__push_b_priors_lognormal()

    def __push_group_priors(self):
        channels = self.__get_channels()
        group_priors = {}
        for channel in channels:
            ela, rate = (self.__priors_group_values[(channel, 'ela')], self.__priors_group_values[(channel, 'rate')])
            # ela
            ela_a_logit = log(ela / (1 - ela))
            ela_b = abs(ela_a_logit * self.__sd)
            # rate
            rate_a_logit = log(rate / (1 - rate))
            rate_b = abs(rate_a_logit * self.__sd)
            # set group priors
            group_priors[(channel, 'ela', 'a')] = (self.__run_id, channel, 'ela', 'a', 'normal', ela_a_logit, ela_b)
            group_priors[(channel, 'ela', 'b')] = (self.__run_id, channel, 'ela', 'b', 'half_normal', ela_b, 'NULL')
            group_priors[(channel, 'rate', 'a')] = (self.__run_id, channel, 'rate', 'a', 'normal', rate_a_logit, rate_b)
            group_priors[(channel, 'rate', 'b')] = (self.__run_id, channel, 'rate', 'b', 'half_normal', rate_b, 'NULL')
        EntityGateway.push_priors_group(self.__run_id, group_priors)

    def __push_b_priors_lognormal(self):
        b_priors_normal = self.__get_b_priors_normal()
        for key in b_priors_normal:
            _list = list(b_priors_normal[key])  # tuple to list in order to overwrite
            _list[4] = 'lognormal'
            mean = b_priors_normal[key][5]
            ia = key.split("_in", 1)[0]
            flexibility = EntityGateway.get_b_prior_flexibility(self.__run_id, ia)
            std = mean * flexibility
            _list[5] = log(mean) - 0.5 * log((std / mean) ** 2 + 1)
            _list[6] = sqrt(log((std / mean) ** 2 + 1))
            b_priors_normal[key] = tuple(_list)
        EntityGateway.push_priors_b(self.__run_id, b_priors_normal)

    def __get_b_priors_normal(self):
        submodels = EntityGateway.get_distinct_list_of_submodels(self.__run_id)
        ias = EntityGateway.get_interaction_counts(self.__run_id)
        b_priors_normal = {}
        for submodel in submodels:
            observed = Observed(self.__run_id, submodel, self.__dates, False, self.__observed_type)
            components = self.__get_submodel_components(submodel)
            if self.__attribution_type == Attribution_Type.tor:
                ia_total = sum([tor_value for sm, ia, _, tor_value in ias if sm == submodel and ia in components])
                component_weights = [(ia, tor_value / ia_total) for sm, ia, _, tor_value in ias if sm == submodel and ia in components]
            else:
                ia_total = sum([last_touch_count for sm, ia, last_touch_count, _ in ias if sm == submodel and ia in components])
                component_weights = [(ia, last_touch_count / ia_total) for sm, ia, last_touch_count, _ in ias if sm == submodel and ia in components]
            # main effect priors
            main_weight = float([(ia, weight) for ia, weight in component_weights if ia == submodel][0][1])  # weight of list index 0
            main_effect = MainEffect(self.__run_id, submodel, self.__dates, include_we=False, set_prior_b=False)
            ela, rate = (self.__priors_group_values[(submodel, 'ela')], self.__priors_group_values[(submodel, 'rate')])
            main_effect.set_as_fitted(0, rate, ela)
            prior = (main_weight * mean(observed.values)) / mean(main_effect.sadstocks)
            b_priors_normal[submodel + '_in_' + submodel] = (self.__run_id, submodel, submodel, 'main', 'normal', prior, prior * self.__sd)
            # ia priors
            for cpn, weight in [v for v in component_weights if v[0] != submodel]:
                ia = Interaction(self.__run_id, main_effect, cpn, self.__dates, False, set_prior_b=False)
                ela, rate = (self.__priors_group_values[(cpn, 'ela')], self.__priors_group_values[(cpn, 'rate')])
                ia.set_as_fitted(0, rate, ela, main_effect)
                prior = float(weight) * mean(observed.values) / (mean(main_effect.sadstocks) * mean(ia.sadstocks))
                b_priors_normal[cpn + '_in_' + submodel] = (self.__run_id, submodel, cpn, 'ia', 'normal', prior, prior * self.__sd)
        return b_priors_normal

    def __get_submodel_components(self, submodel):
        return [v[0] for v in EntityGateway.get_model_components(self.__run_id, submodel)]

    def __get_channels(self):
        channels = []
        submodels = EntityGateway.get_distinct_list_of_submodels(self.__run_id)
        for submodel in submodels:
            channels += self.__get_submodel_components(submodel)
        return set(channels)
