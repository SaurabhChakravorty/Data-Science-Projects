from numpy import array

from db_interaction.entity_gateway import EntityGateway
from modeling.component_container import Prior
from modeling.independent import Independent
from prior_factory.distribution_provider import DistributionProvider, Distribution_Type

"""
[i] set_prior_b option in __init__ due to prior_factory where this object is used to determine the prior

To dos:
"""


class MainEffect(Independent):

    def __init__(self, run_id, channel_name, dates, include_we, set_prior_b=True):
        super().__init__(run_id, channel_name, dates, include_we)
        self.__name = channel_name
        self.__result_values = None
        if set_prior_b:
            self.__prior_b = self._get_prior_b(run_id)

    # region properties

    @property
    def channel_name(self):
        return self.__name

    @property
    def prior_b(self):
        return self.__prior_b

    @property
    def name(self):
        return 'main_' + self.__name

    @property
    def rate_name(self):
        return 'main_' + self.__name + '_rate'

    @property
    def ela_name(self):
        return 'main_' + self.__name + '_ela'

    # endregion

    # region result properties

    @property
    def result_values(self):
        if self.is_fitted == True:
            return array([float(self._b * v) for v in self.sadstocks])
        else:
            raise Exception('Cannot provide results as ' + self.__name + ' is not fitted.')

    # endregion

    def set_as_fitted(self, beta, rate, ela, interactor=None):
        self._set_as_fitted(beta, rate, ela)

    def set_prior_b(self, run_id):
        self.__prior_b = self._get_prior_b(run_id)

    def _get_prior_b(self, run_id):
        result = EntityGateway.get_prior_b(run_id, self.__name, self.__name)
        name = 'main_' + self.__name + '_b'
        distribution = DistributionProvider.instance().get_pm_distribution(Distribution_Type[result[0]])
        return Prior(name, distribution, result[1], result[2])
