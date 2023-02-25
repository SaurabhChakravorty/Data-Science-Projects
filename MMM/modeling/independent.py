from abc import ABC, abstractmethod

from numpy import array
from math import pow
from numpy import float64 as flt
from theano import tensor as tt
from theano.ifelse import ifelse

from db_interaction.entity_gateway import EntityGateway

"""
* One single underscore marks protected method.

To dos:
* [!] Need to use tt.ifelse in tensor vector below but don't know why. Enforces necessary update? 
  https://stackoverflow.com/questions/33968233/how-to-perform-conditional-updates-on-shared-variables-in-theano
"""


class Independent(ABC):

    def __init__(self, run_id, name, dates, include_we):
        self.__is_fitted = False
        self.__name = name
        values = EntityGateway.get_channel_values(run_id, self.__name, 'value', dates[0], dates[1], include_we)
        self.__min = min(values)
        self.__max = max(values)
        self.__values = self.__get_rescaled_predictors(values)

        # [i] set after fitting process with def set_as_fitted()
        self.__b = None
        self.__rate = None
        self.__ela = None
        self.__sadstocks = None

    def __iter__(self):
        return (i for i in self.__values)

    # region names

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def channel_name(self):
        pass

    @property
    @abstractmethod
    def prior_b(self):
        pass

    @property
    @abstractmethod
    def rate_name(self):
        pass

    @property
    @abstractmethod
    def ela_name(self):
        pass

    # endregion

    @property
    def _b(self):
        return self.__b

    @property
    def is_fitted(self):
        return self.__is_fitted

    @property
    def values(self):
        return array(self.__values)

    @property
    def sadstocks(self):
        if self.is_fitted == True:
            return array(self.__sadstocks)
        else:
            raise Exception('Cannot provide results as ' + self.__name + ' is not fitted.')

    @abstractmethod
    def set_as_fitted(self, beta, rate, ela, interactor=None):
        pass

    @abstractmethod
    def _get_prior_b(self, run_id):
        pass

    def overwrite_values(self, values, allow_length_change=False):
        if allow_length_change is False and len(values) != len(self.__values):
            raise Exception('Cannot replace values of independent. Arrays do not have the same length')
        self.__values = self.__get_rescaled_predictors(values)
        self.__sadstocks = self.get_sadstocks(self.__rate, self.__ela)

    def _set_as_fitted(self, b, rate, ela):
        if b < 0:
            raise ValueError("Conversion coefficient (b) must be greater than zero")
        if rate < 0 or rate > 1:
            raise ValueError("Rate must be greater than zero and smaller than or equal to one")
        if ela < 0 or ela > 1:
            raise ValueError("Elasticity must be greater than zero and smaller than or equal to one")

        self.__b = b
        self.__rate = rate
        self.__ela = ela
        self.__sadstocks = self.get_sadstocks(rate, ela)
        self.__is_fitted = True

    def __get_rescaled_predictors(self, values, overwrite_minmax=False):
        if overwrite_minmax:
            self.__min = min(values)
            self.__max = max(values)
        _values = [v if v >= self.__min else self.__min for v in values]
        spread = self.__max - self.__min
        if spread == 0:
            return [round(v, 2) for v in _values]
        else:
            return [round((((v - self.__min) / spread) * 999 + 1), 2) for v in _values]

    def get_sadstocks(self, rate, ela):
        return [pow(v, ela) for v in self.__get_adstocks(self.__values[:], rate)]  # [i] pass shallow copy, otherwise list is overwritten

    def __get_adstocks(self, l, rate, i=0):
        if i + 1 < len(l):
            l[i + 1] += rate * l[i]
            self.__get_adstocks(l, rate, i + 1)
        return l

    # region tensor values

    def get_sadstock_tensor(self, rate, ela):
        adstock_tensor = self.__get_adstock_tensor_vector(rate)
        return self.__get_sadstock_tensor_vector(ela, adstock_tensor)

    def __get_adstock_tensor_vector(self, rate):
        adstocks_tensor = []
        for i, v in enumerate(self.__values):
            _v = flt(v)
            zero = flt(0)
            if i == 0:
                adstocks_tensor.append(_v)
            else:
                adstocks_tensor.append(ifelse(array(True), tt.add(_v, tt.mul(rate, adstocks_tensor[-1])), zero))
        return adstocks_tensor

    def __get_sadstock_tensor_vector(self, ela, adstock_tensor):
        sadstock_tensor = []
        for v in adstock_tensor:
            sadstock_tensor.append(tt.pow(v, ela))
        return sadstock_tensor

    # endregion
