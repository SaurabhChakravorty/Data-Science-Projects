from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.stats as st
from scipy.special import logit

from singleton_pattern import Singleton


class Distribution_Type(Enum):
    gamma = 'gamma'
    beta = 'beta'
    normal = 'normal'
    lognormal = 'lognormal'
    logitnormal = 'logitnormal'
    half_normal = 'half_normal'
    positive_normal = 'positive_normal'
    zero_one_normal = 'zero_one_normal'
    uniform = 'uniform'
    cauchy = 'cauchy'


"""
To dos:
* Add more distribution to get_pdf
"""


@Singleton
class DistributionProvider:
    LOWER_BOUND = float(0.000005)
    UPPER_BOUND = float(0.999995)

    def get_pm_distribution(self, distribution_type):
        if not isinstance(distribution_type, Distribution_Type):
            raise TypeError('distribution_type must be an instance of Distribution_Type Enum')

        if distribution_type == Distribution_Type.beta:
            return pm.Beta
        elif distribution_type == Distribution_Type.gamma:
            return pm.Gamma
        elif distribution_type == Distribution_Type.normal:
            return pm.Normal
        elif distribution_type == Distribution_Type.half_normal:
            return pm.HalfNormal
        elif distribution_type == Distribution_Type.lognormal:
            return pm.Lognormal
        elif distribution_type == Distribution_Type.logitnormal:
            return pm.LogitNormal
        elif distribution_type == Distribution_Type.uniform:
            return pm.Uniform
        elif distribution_type == Distribution_Type.cauchy:
            return pm.Cauchy
        elif distribution_type == Distribution_Type.positive_normal:
            return pm.Bound(pm.Normal, lower=self.LOWER_BOUND)
        elif distribution_type == Distribution_Type.zero_one_normal:
            return pm.Bound(pm.Normal, lower=self.LOWER_BOUND, upper=self.UPPER_BOUND)
        else:
            raise Exception("[!] get_pm_distribution error: no matching PyMC3 distribution found")

    def get_pdf(self, distribution_type, a, b, show=False):
        if distribution_type == Distribution_Type.logitnormal:
            x = np.linspace(0.0001, 0.9999, 500)
            pdf = st.norm.pdf(logit(x), loc=a, scale=b) * 1 / (x * (1 - x))
            if show:
                plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(a, b))
                plt.legend(loc=1)
                plt.show()
            return x, pdf
        else:
            raise Exception("[!] get_pdf error: no matching PyMC3 distribution found")