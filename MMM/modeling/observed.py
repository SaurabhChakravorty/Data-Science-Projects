from numpy import array

from db_interaction.entity_gateway import EntityGateway
from modeling.enums import Observed_Type

"""
To dos: None

Notes:
* [!] If value <= 0 then 0.0001
"""


class Observed:

    def __init__(self, run_id, submodel, dates, include_we, observed_type=Observed_Type.lc_orders):
        if not isinstance(observed_type, Observed_Type):
            raise TypeError('dependent_type must be an instance of Dependent_Type Enum')

        self.__submodel_name = submodel
        self.__observed_type = observed_type.value
        values = EntityGateway.get_channel_values(run_id, self.__submodel_name, self.__observed_type, dates[0], dates[1], include_we)
        self.__values = self.__get_values_greater_zero(values)  # not applying leads to 'bad initial energy as lognormal can not become 0'

    def __iter__(self):
        return (i for i in self.__values)

    @property
    def name(self):
        return self.__submodel_name + '_' + self.__observed_type

    @property
    def submodel_name(self):
        return self.__submodel_name

    @property
    def values(self):
        return array(self.__values)

    def __get_values_greater_zero(self, values):
        return [v if v > 0 else 0.001 for v in values]

    def overwrite_values(self, values):
        self.__values = self.__get_values_greater_zero(values)
