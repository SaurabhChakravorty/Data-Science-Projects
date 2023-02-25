from numpy import array

from db_interaction.entity_gateway import EntityGateway
from modeling.component_container import Prior
from modeling.independent import Independent
from modeling.main_effect import MainEffect
from prior_factory.distribution_provider import DistributionProvider, Distribution_Type

"""
[i] set_prior_b option in __init__ due to prior_factory where this object is used to determine the prior

To dos:
* Overriding the main effect in set_as_fitted() might not be necessary because of reference type.
"""


class Interaction(Independent):

    def __init__(self, run_id, main_effect, channel_name, dates, include_we, set_prior_b=True):
        super().__init__(run_id, channel_name, dates, include_we)
        self.__name = channel_name

        # set or overwritten after fitting
        if not isinstance(main_effect, MainEffect):
            raise TypeError("Interactor of interaction must be of type MainEffect")
        if len(self.values) != len(main_effect.values):
            raise ValueError("Input vectors must be of the same size")
        self.__main_effect = main_effect
        self.__result_values = None
        if set_prior_b:
            self.__prior_b = self._get_prior_b(run_id)  # [i] put here due to dependency with main_effect

    # region names

    @property
    def channel_name(self):
        return self.__name

    @property
    def prior_b(self):
        return self.__prior_b

    @property
    def name(self):
        return 'ia_' + self.__name

    @property
    def rate_name(self):
        return 'ia_' + self.__name + '_in_' + self.__main_effect.channel_name + '_rate'

    @property
    def ela_name(self):
        return 'ia_' + self.__name + '_in_' + self.__main_effect.channel_name + '_ela'

    # endregion

    # region result properties

    @property
    def result_values(self):
        if self.is_fitted == True:
            return array([float(self._b * this * main) for this, main in zip(self.sadstocks, self.__main_effect.sadstocks)])
        else:
            raise Exception('Cannot provide results as ' + self.__name + ' is not fitted.')

    # endregion

    def set_as_fitted(self, beta, rate, ela, main_effect=None):
        if not isinstance(main_effect, MainEffect):
            raise TypeError("Interactor of interaction must be of type MainEffect")
        if len(self.values) != len(main_effect.values):
            raise ValueError("Input vectors must be of the same size")
        self.__main_effect = main_effect  # [!] necessary?
        self._set_as_fitted(beta, rate, ela)

    def set_prior_b(self, run_id):
        self.__prior_b = self._get_prior_b(run_id)

    def _get_prior_b(self, run_id):
        result = EntityGateway.get_prior_b(run_id, self.__main_effect.channel_name, self.__name)
        name = 'ia_' + self.__name + '_in_' + self.__main_effect.channel_name + '_b'
        distribution = DistributionProvider.instance().get_pm_distribution(Distribution_Type[result[0]])
        return Prior(name, distribution, result[1], result[2])
