from abc import ABC, abstractmethod
from collections import namedtuple

from modeling.component_container import ComponentContainer

DistParas = namedtuple('DistParas', 'a b')

"""
To dos:
* Save az.from_pymc3(self.trace) along with the trace for better az.summary performance: 
  https://discourse.pymc.io/t/compute-reduced-summary-statistics/4971/2
* Lognormal by default; change that
* As a parameter? discard_tuned_samples=False in self.__trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores)
"""


class QIAmmm(ABC):

    def __init__(self, run_id):
        # private
        self.__submodels = {}
        # [i] set when fitted
        self.__name = 'no_name'
        self.__cc = None
        # protected
        self._run_id = run_id
        self._pm_model = None
        self._trace = None
        self._sampling_args = (100, 100, 2, 1)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def run_id(self):
        return self._run_id

    @property
    def component_container(self):
        return self.__cc

    @property
    def trace(self):
        return self._trace

    @property
    def pm_model(self):
        return self._pm_model

    def add_submodel(self, submodel):
        self.__submodels[submodel.dependent.name] = submodel

    def set_sampling_args(self, draws, tune, chains, cores):
        self._sampling_args = (draws, tune, chains, cores)

    def run(self):
        if (any(self.__submodels)):  # [i] returns True if d contains at least one truelike key
            self.__set_cc()
            self._fit()
        else:
            raise Exception('Cannot fit model as model is empty (no submodels).')

    def close(self):
        if (any(self.__submodels)):  # [i] returns True if d contains at least one truelike key
            self.__set_cc()
        else:
            raise Exception('Cannot fit model as model is empty (no submodels).')

    def _set_prior_bs(self):
        self.__cc.set_prior_bs(self._run_id)

    def __set_cc(self):
        self.__cc = ComponentContainer(self._run_id, list(self.__submodels.values()))  # [i] list() to avoid pickling error

    @abstractmethod
    def _fit(self):
        pass

    @abstractmethod
    def set_as_fitted(self):
        pass
