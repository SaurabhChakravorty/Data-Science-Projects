from enum import Enum

"""
* The allocation is determined by the first independent added
* Allows to set critical properties after instantiation [risk]

To dos:
* Introduce more allocation options
* New child 'InvestmentChannel'?
"""


class Investment_Type(Enum):
    const = 'const'
    flex = 'flex'


class Channel:

    def __init__(self, name, cpa_base=None, spend_base=None, investment_type=None):
        # mandatory attributes
        self.__name = name
        self.__spend_base = spend_base
        self.__cpa_base = cpa_base
        self.__investment_type = investment_type
        # dynamic attributes
        self.__spend_new = self.__spend_base
        self.__cpa_new = self.__cpa_base
        self.__min = 0
        self.__max = None
        self.__spend_distribution = None  # [i] spend_distribution is set along with the first independent added
        self.__independents = {}

    # region properties

    @property
    def name(self):
        return self.__name

    @property
    def spend_base(self):
        return self.__spend_base

    @spend_base.setter
    def spend_base(self, spend_base):
        if self.__spend_base is None:
            self.__spend_base = spend_base
        self.__spend_base = spend_base

    @property
    def cpa_base(self):
        return self.__cpa_base

    @cpa_base.setter
    def cpa_base(self, cpa_base):
        if self.__cpa_new is None:
            self.__cpa_new = cpa_base
        self.__cpa_base = cpa_base

    @property
    def investment_type(self):
        return self.__investment_type

    @investment_type.setter
    def investment_type(self, investment_type):
        if not isinstance(investment_type, Investment_Type):
            raise TypeError("[!] 'inv_type' of Channel must be of type Investment_Type [enum]")
        self.__investment_type = investment_type

    @property
    def spend_new(self):
        return self.__spend_new

    @spend_new.setter
    def spend_new(self, spend_new):
        self.__spend_new = spend_new

    @property
    def cpa_new(self):
        return self.__cpa_new

    @cpa_new.setter
    def cpa_new(self, cpa_new):
        self.__cpa_new = cpa_new

    @property
    def min(self):
        return self.__min

    @min.setter
    def min(self, min):
        self.__min = min

    @property
    def max(self):
        return self.__max

    @max.setter
    def max(self, max):
        self.__max = max

    @property
    def allocation(self):
        return self.__spend_distribution

    @allocation.setter
    def allocation(self, allocation):
        self.__spend_distribution = allocation

    # endregion

    def add_independent(self, independent, name):
        if bool(self.__independents) is False:
            self.__spend_distribution = self.__get_relative_investment_per_day(independent)
        if name in self.__independents.keys():
            raise NameError("[!] Error: The independent you are trying to add does already exist")
        self.__independents[name] = independent

    def change_spend(self, new_spend, ignore_inv_type=False):
        if new_spend >= 0:
            self.__spend_new = new_spend
        else:
            self.__spend_new = 0
        if self.__spend_base != 0 and (ignore_inv_type == True or self.__investment_type != Investment_Type.const):
            for independent in self.__independents.values():
                predictor_total = new_spend / self.__cpa_base
                independent.overwrite_values([predictor_total * v for v in self.__spend_distribution])

    def change_values(self, values):
        for independent in self.__independents.values():
            independent.overwrite_values(values, allow_length_change=True)

    def __get_relative_investment_per_day(self, independent):
        min = independent._Independent__min
        max = independent._Independent__max
        spread = max - min
        values = [((v - 1) * spread) / 999 + min for v in independent]
        total = sum(values)
        return [i / total for i in values]