import datetime
from collections import namedtuple

from numpy import array
from pandas import DataFrame

from db_interaction.entity_gateway import EntityGateway
from modeling.component_container import Prior
from modeling.enums import Observed_Type, Control_Type
from modeling.interaction import Interaction
from modeling.main_effect import MainEffect
from modeling.observed import Observed
from prior_factory.distribution_provider import DistributionProvider, Distribution_Type

"""
To dos:
* Integrate control priors != dummies better. Control as an own class?
"""


Dummy = namedtuple('Dummy', 'name prior values')


class Submodel:

    def __init__(self, run_id, channel_name, dates, include_we, observed_type=Observed_Type.lc_orders):
        self.__start_and_end_date = dates
        self.__is_fitted = False
        self.__name = channel_name
        self.__prior_sd = 1  # [i] default, not guaranteed to work; set with prior predictive sampler
        # channels
        cpns = EntityGateway.get_model_components(run_id, channel_name)
        main_channel = [cpn[0] for cpn in cpns if cpn[1] == 'main'][0]
        self.__observed = Observed(run_id=run_id, submodel=main_channel, dates=dates, include_we=include_we, observed_type=observed_type)
        self.__main_effect = MainEffect(run_id=run_id, channel_name=main_channel, dates=dates, include_we=include_we)
        ia_names = [cpn[0] for cpn in cpns if cpn[1] == 'ia']
        self.__interactions = [Interaction(run_id, self.__main_effect, name, dates, include_we) for name in ia_names]
        # dates, trend, control values
        self.__dates = self.get_date_list(dates, run_id, include_we)
        self.__trend_values = [float(i + 1) for i, v in enumerate(self.__dates)]
        self.__we_values = [float(1) if date.weekday() >= 5 else float(0) for date in self.__dates]
        # dummies
        self.__dummies = self.__get_dummies(run_id)
        # control priors: [!] dummies not included
        self.__control_priors = self.__get_control_prior_dict(run_id)
        # [i] set when is_fitted = true
        self.__control_posteriors = {Control_Type.trend.value: None, Control_Type.we.value: None, Control_Type.intercept.value: None}
        for dummy in self.__dummies:
            self.__control_posteriors[dummy.name] = None
        # [!] Instantiate df after all other attributes due to dependencies
        self.__df = self.__get_df()

    # region properties

    @property
    def name(self):
        return self.__name

    @property
    def start_and_end_date(self):
        return self.__start_and_end_date

    @property
    def prior_sd(self):
        return self.__prior_sd

    @prior_sd.setter
    def prior_sd(self, value):
        self.__prior_sd = value

    @property
    def is_fitted(self):
        return self.__is_fitted

    @property
    def dependent(self):
        return self.__observed

    @property
    def main_effect(self):
        return self.__main_effect

    @property
    def interactions(self):
        return self.__interactions

    @property
    def control_priors(self):
        return self.__control_priors

    @property
    def dates(self):
        return self.__dates

    @property
    def dummies(self):
        return self.__dummies

    @property
    def trend_values(self):
        return array(self.__trend_values)

    @property
    def we_values(self):
        return array(self.__we_values)

    @property
    def df(self):
        return self.__df

    # region posteriors

    @property
    def result_values(self):
        if self.is_fitted == True:
            return self.get_result_values()
        else:
            raise Exception('Cannot provide result values as ' + self.__observed.name + ' is not fitted.')

    @property
    def intercept_result_values(self):
        if self.is_fitted == True:
            return self.__get_intercept_result_values()
        else:
            raise Exception('Cannot provide intercept result values as ' + self.__observed.name + ' is not fitted.')

    @property
    def trend_result_values(self):
        if self.is_fitted == True:
            return self.__get_trend_result_values()
        else:
            raise Exception('Cannot provide trend result values as ' + self.__observed.name + ' is not fitted.')

    @property
    def weekend_result_values(self):
        if self.is_fitted == True:
            return self.__get_weekend_result_values()
        else:
            raise Exception('Cannot provide weekend result_values as ' + self.__observed.name + ' is not fitted.')

    @property
    def intercept_posterior(self):
        if self.is_fitted == True:
            return self.__control_posteriors[Control_Type.intercept.value]
        else:
            raise Exception('Cannot provide intercept posterior as ' + self.__observed.name + ' is not fitted.')

    @property
    def trend_posterior(self):
        if self.is_fitted == True:
            return self.__control_posteriors[Control_Type.trend.value]
        else:
            raise Exception('Cannot provide trend posterior as ' + self.__observed.name + ' is not fitted.')

    @property
    def we_posterior(self):
        if self.is_fitted == True:
            return self.__control_posteriors[Control_Type.we.value]
        else:
            raise Exception('Cannot provide weekend posterior as ' + self.__observed.name + ' is not fitted.')

    @property
    def dummy_posteriors(self):
        if self.is_fitted == True:
            return {k: v for k, v in self.__control_posteriors.items() if k.startswith('dummy')}
        else:
            raise Exception('Cannot provide dummy posterior as ' + self.__observed.name + ' is not fitted.')

    # endregion

    # endregion

    @staticmethod
    def get_date_list(dates, run_id, include_we):
        start = datetime.datetime.strptime(dates[0], '%Y-%m-%d')
        end = datetime.datetime.strptime(dates[1], '%Y-%m-%d')
        exclusions = EntityGateway.get_days_to_exclude(run_id)
        date_list = sorted([end - datetime.timedelta(days=x) for x in range((end - start).days + 1) if end - datetime.timedelta(days=x) not in exclusions])
        if include_we:
            return date_list
        else:
            return [date for date in date_list if date.weekday() < 5]

    def set_prior_bs(self, run_id):
        self.__main_effect.set_prior_b(run_id)
        for interaction in self.__interactions:
            interaction.set_prior_b(run_id)

    def overwrite_dates(self, dates, run_id, include_we):
        self.__dates = self.get_date_list(dates, run_id, include_we)

    def set_as_fitted(self, means):
        # intercept
        if Control_Type.intercept.value + '_' + self.name in means:
            self.__control_posteriors[Control_Type.intercept.value] = means[Control_Type.intercept.value + '_' + self.name]
        else:
            self.__control_posteriors[Control_Type.intercept.value] = None
        # trend
        if Control_Type.trend.value + '_' + self.name in means:
            self.__control_posteriors[Control_Type.trend.value] = means[Control_Type.trend.value + '_' + self.name]
        else:
            self.__control_posteriors[Control_Type.trend.value] = None
        # weekend
        if Control_Type.we.value + '_' + self.name in means:
            self.__control_posteriors[Control_Type.we.value] = means[Control_Type.we.value + '_' + self.name]
        else:
            self.__control_posteriors[Control_Type.we.value] = None
        # dummies
        for d in self.__dummies:
            if d.name in means:
                self.__control_posteriors[d.name] = means[d.name]
            else:
                self.__control_posteriors[d.name] = None
        # main effect
        self.__main_effect.set_as_fitted(means[self.__main_effect.prior_b.name], means[self.__main_effect.rate_name], means[self.__main_effect.ela_name])
        # interactions
        for ia in self.__interactions:
            ia.set_as_fitted(means[ia.prior_b.name], means[ia.rate_name], means[ia.ela_name], self.__main_effect)
        # is fitted flag
        self.__is_fitted = True

    def get_result_values(self):
        itc = self.__get_intercept_result_values()
        trend = self.__get_trend_result_values()
        we = self.__get_weekend_result_values()
        dummies = [self.get_dummy_result_values(d) for d in self.__dummies]
        ia_result_values = [ia.result_values for ia in self.__interactions]
        return array([sum(v) for v in zip(itc, trend, we, self.__main_effect.result_values, *dummies, *ia_result_values)])

    def __get_intercept_result_values(self):
        if self.__control_posteriors[Control_Type.intercept.value] is not None:
            return [float(self.__control_posteriors[Control_Type.intercept.value]) for _ in range(0, len(self.__observed.values))]
        else:
            return [0. for _ in range(0, len(self.__observed.values))]

    def __get_trend_result_values(self):
        if self.__control_posteriors[Control_Type.trend.value] is not None:
            return [float(self.__control_posteriors[Control_Type.trend.value] * v) for v in self.__trend_values]
        else:
            return [0. for _ in range(0, len(self.__observed.values))]

    def __get_weekend_result_values(self):
        if self.__control_posteriors[Control_Type.we.value] is not None:
            return [float(self.__control_posteriors[Control_Type.we.value] * v) for v in self.__we_values]
        else:
            return [0. for _ in range(0, len(self.__observed.values))]

    def get_dummy_result_values(self, dummy):
        if self.__control_posteriors[dummy.name] is not None:
            return [float(self.__control_posteriors[dummy.name] * v) for v in dummy.values]
        else:
            return [0. for _ in range(0, len(self.__observed.values))]

    def __get_control_prior_dict(self, run_id):
        control_priors = {}
        # trend
        trend = self.__get_control_prior(run_id, Control_Type.trend)
        if trend is not None:
            control_priors[Control_Type.trend.value] = trend
        # intercept
        intercept = self.__get_control_prior(run_id, Control_Type.intercept)
        if intercept is not None:
            control_priors[Control_Type.intercept.value] = intercept
        # weekend
        we = self.__get_control_prior(run_id, Control_Type.we)
        if we is not None:
            control_priors[Control_Type.we.value] = we
        return control_priors

    def __get_control_prior(self, run_id, control_type, dummy_name=None):
        prior_args = EntityGateway.get_control_prior(run_id, control_type.value)
        if prior_args is not None:
            distribution = DistributionProvider.instance().get_pm_distribution(Distribution_Type[prior_args[0]])
            if dummy_name is not None:
                name = dummy_name
            else:
                name = control_type.value
            return Prior(name, distribution, prior_args[1], prior_args[2])
        else:
            return None

    def __get_dummies(self, run_id):
        if self.__get_control_prior(run_id, Control_Type.dummy) is None:
            return []
        dummies = []
        dummy_dates = EntityGateway.get_dummy_dates(run_id)
        for dummy_date in dummy_dates:
            date_formatted = datetime.datetime.strftime(dummy_date, '%Y-%m-%d')
            name = 'dummy_' + self.__name + '_' + datetime.datetime.strftime(dummy_date, '%Y-%m-%d')
            values = array([1 if date_formatted == datetime.datetime.strftime(date.date(), '%Y-%m-%d') else 0 for date in self.__dates])
            prior = self.__get_control_prior(run_id, Control_Type.dummy)
            dummies.append(Dummy(name, prior, values))
        return dummies

    def __get_df(self):
        df = DataFrame()
        df['dates'] = self.__dates
        df[self.__main_effect.name] = self.__main_effect.values
        for interaction in self.__interactions:
            df[interaction.name] = interaction.values
        df[self.__observed.name] = self.__observed.values
        if self.__trend_values is not None:
            df[Control_Type.trend.value] = self.__trend_values
        if self.__we_values is not None:
            df[Control_Type.we.value] = self.__we_values
        for d in self.__dummies:
            df[d.name] = d.values
        return df