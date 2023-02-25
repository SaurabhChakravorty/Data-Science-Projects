from collections import namedtuple

from db_interaction.entity_gateway import EntityGateway
from prior_factory.distribution_provider import DistributionProvider, Distribution_Type

Prior = namedtuple('Prior', 'name dist a b')

"""
To dos:
* Implement check if inputs are correct
"""


class ComponentContainer:

    def __init__(self, run_id, submodels):
        self.__submodels = submodels
        self.__range = range(0, len(self.__submodels[0].dependent.values))
        self.__channel_names = self.get_channel_names()
        # priors
        self.__group_priors = self.__get_group_priors(run_id)
        self.__derived_priors = self.__get_derived_priors(run_id)
        self.__sigma_prior = self.__get_sigma_prior(run_id)
        self.__priors = self.__get_prior_dict()

    def __iter__(self):
        return (i for i in self.__submodels)

    def __len__(self):
        return len(self.__submodels.values())

    # region properties

    @property
    def range(self):
        return self.__range

    @property
    def submodels(self):
        return self.__submodels

    @property
    def channel_names(self):
        return self.__channel_names

    @property
    def priors(self):
        return self.__priors

    @property
    def group_priors(self):
        return self.__group_priors

    @property
    def derived_priors(self):
        return self.__derived_priors

    @property
    def sigma_prior(self):
        return self.__sigma_prior

    # endregion

    def get_channel_names(self):
        names = []
        for sm in self.__submodels:
            names.append(sm.main_effect.channel_name)
            for ia in sm.interactions:
                names.append(ia.channel_name)
        return set(names)

    def set_prior_bs(self, run_id):
        for sm in self.__submodels:
            sm.set_prior_bs(run_id)

    def __get_group_priors(self, run_id):
        priors_group = {}
        for name in self.__channel_names:
            priors_group_args = EntityGateway.get_group_priors(run_id, name)
            priors = {}
            for value in priors_group_args:
                sub_name = value[0] + '_group_' + value[1] + '_' + value[2]
                distribution = DistributionProvider.instance().get_pm_distribution(Distribution_Type[value[3]])
                priors[value[1] + '_' + value[2]] = Prior(sub_name, distribution, value[4], value[5])
            priors_group[name] = priors
        return priors_group

    def __get_derived_priors(self, run_id):
        priors_derived = {}
        priors_derived_args = EntityGateway.get_derived_priors(run_id)
        for value in priors_derived_args:
            distribution = DistributionProvider.instance().get_pm_distribution(Distribution_Type[value[1]])
            priors_derived[value[0]] = Prior(None, distribution, None, None)
        return priors_derived

    def __get_sigma_prior(self, run_id):
        prior_sigma_args = EntityGateway.get_sigma(run_id)
        distribution = DistributionProvider.instance().get_pm_distribution(Distribution_Type[prior_sigma_args[0]])
        return Prior('sigma', distribution, prior_sigma_args[1], prior_sigma_args[2])

    def __get_prior_dict(self):
        # [i] derived priors are equal to group priors
        priors = {}
        # sigma
        priors['sigma'] = self.__sigma_prior
        # control priors
        for control in self.__submodels[0].control_priors.values():
            priors[control.name] = control
        # rate and ela group priors
        for channel in self.__group_priors.values():
            for gp in channel.values():
                priors[gp.name] = gp
        # main effects and interactions
        for sm in self.__submodels:
            priors[sm.main_effect.prior_b.name] = sm.main_effect.prior_b
            for ia in sm.interactions:
                priors[ia.prior_b.name] = ia.prior_b
        return priors
