import sys
import threading
from datetime import datetime

from db_interaction import push
from db_interaction.entity_gateway import EntityGateway
from evaluation.result_reporter import ResultReporter
from modeling.observed import Observed_Type
from modeling.qiammm_log_const import LogConstMMM
from modeling.qiammm_log_logit import LogLogitMMM
from modeling.submodel import Submodel
from fitting.platform_manager import PlatformManager
from prior_factory.log_logit import Attribution_Type, LogLogit
from prior_factory.prior_predictive_sampler import PPS_LogLogit
from scenario_planning.value_change_scenario import ValueChangeScenario
from input_scanning.input_scanner import InputScanner

"""
Notes:
* Wrapper for the regression classes 
* Takes care of the model fitting and pushing
* Use to run different model configurations

To dos:
"""


class Fitter:

    def __init__(self):
        threading.stack_size(PlatformManager.instance().stack_size)
        sys.setrecursionlimit(PlatformManager.instance().recursion_limit)
        # [i] simple attribute implementation as low-risk and high-level code
        self.draws = 50
        self.tune = 50
        self.chains = 2
        self.cores = 1

    def fit_and_push_log_const(self, model_name, run_id, sd, dates, observed_type, attribution_type, fit_all_coefs=True, test_dates=None, include_we=False):
        # run input checks
        self.__run_input_checks(model_name, observed_type)
        input_scanner = InputScanner(run_id, observed_type, dates, test_dates, include_we)
        input_scanner.run_input_scanner()
        # calculate priors
        bp = LogLogit(run_id=run_id, sd=sd, dates=dates, attribution_type=attribution_type, observed_type=observed_type)
        bp.run()
        # initialize model
        model = LogConstMMM(run_id, push.Pusher.get_new_push_id(), model_name)
        model.fit_all_coefs = fit_all_coefs
        model.name = model_name
        model.set_sampling_args(self.draws, self.tune, self.chains, self.cores)
        EntityGateway.update_history_tables(run_id=run_id, push_id=model.push_id, attribution=Attribution_Type.tor,
                                            model_name=model_name)
        # build submodels
        submodel_names = EntityGateway.get_distinct_list_of_submodels(run_id)
        for name in submodel_names:
            sm = Submodel(run_id, name, dates, include_we=include_we, observed_type=observed_type)
            model.add_submodel(sm)
        # run
        print('[i] Starting model run...')
        start_time = datetime.now()
        model.run()
        return self.__push_results(model, model_name, run_time=datetime.now() - start_time, run_id=run_id,
                                   push_id=model.push_id, dates=dates, test_dates=test_dates, include_we=include_we)

    def fit_and_push_log_logit(self, model_name, run_id, sd, dates, observed_type, attribution_type, test_dates=None, include_we=False):
        # run input checks
        self.__run_input_checks(model_name, observed_type)
        input_scanner = InputScanner(run_id, observed_type, dates, test_dates, include_we)
        input_scanner.run_input_scanner()
        # calculate priors
        bp = LogLogit(run_id=run_id, sd=sd, dates=dates, attribution_type=attribution_type, observed_type=observed_type)
        bp.run()
        # initialize model
        model = LogLogitMMM(run_id, push.Pusher.get_new_push_id())
        model.name = model_name
        model.set_sampling_args(self.draws, self.tune, self.chains, self.cores)
        EntityGateway.update_history_tables(run_id=run_id, push_id=model.push_id, attribution=attribution_type,
                                            model_name=model_name)
        # get submodel sds
        print('[i] Sampling dependent sds...')
        sds = PPS_LogLogit.instance().get_dependent_sds(
            run_id=run_id,
            dates=dates,
            nb_of_samples=PlatformManager.instance().prior_samples,
            observed_type=Observed_Type.lc_orders,
            include_we=False)
        # build submodels
        submodel_names = EntityGateway.get_distinct_list_of_submodels(run_id)
        for name in submodel_names:
            sm = Submodel(run_id, name, dates, include_we=include_we, observed_type=observed_type)
            sm.prior_sd = float(sds[name])
            model.add_submodel(sm)
        # run
        print('[i] Starting model run...')
        start_time = datetime.now()
        model.run()
        return self.__push_results(model, model_name, run_time=datetime.now() - start_time, run_id=run_id,
                                   push_id=model.push_id, dates=dates, test_dates=test_dates, include_we=include_we)

    def __run_input_checks(self, model_name, observed_type):
        if not isinstance(observed_type, Observed_Type):
            raise TypeError('[!] dependent_type must be an instance of Dependent_Type Enum')
        if len(model_name) > 50:
            raise ValueError('[!] Model name is too long. Maximum length is 50 characters')
        if model_name in EntityGateway.get_distinct_list_of_model_names():
            raise ValueError('[!] Model name already exists. Please define a unique model name to save history tables')

    def __run_input_scanner(self, run_id):
        scn = InputScanner(run_id)
        scn.run_scanner()

    def __push_results(self, model, model_name, run_time, run_id, push_id, dates, test_dates, include_we):
        # initialize diagnostics (train and test) and try set reporter diagnostics for trained model
        waic, r2_regular_train, r2_regular_test, r2_bayesian_train, r2_bayesian_test, r_hat_max, div_nb, div_perc = \
            -1, -1, -1, -1, -1, -1, -1, -1
        try:
            reporter = ResultReporter(model)
            waic = reporter.waic
            r2_regular_train = reporter.r2_regular
            r2_bayesian_train = reporter.r2_bayesian
            r_hat_max = reporter.r_hat_max
            div_nb, div_perc = reporter.get_divergences()
        except:
            print('[!] Error while trying to run Result Reporter for trained model.')
        # push results
        pusher = push.Pusher(model=model, run_id=run_id, push_id=push_id, dates=dates, include_we=include_we, model_score=waic)
        pusher.push_model()
        pusher.push_sampling_results()
        # try set reporter for test r2 (needs to be run after model is pushed to database as to run ValueChangeScenario)
        try:
            if test_dates is not None:
                model_test_set = ValueChangeScenario(push_ids=[push_id])
                model_test_set.run(run_id=run_id, dates=test_dates, include_we=False)
                reporter_test_set = ResultReporter(model_test_set.model, [push_id])
                r2_regular_test = reporter_test_set.r2_regular
                r2_bayesian_test = reporter_test_set.r2_bayesian
        except:
            print('[!] Error while trying to run Result Reporter for trained model.')
        pusher.push_diagnostics(run_time=run_time, waic=waic, r2_regular_train=r2_regular_train, r2_regular_test=r2_regular_test,
                                r2_bayesian_train=r2_bayesian_train, r2_bayesian_test=r2_bayesian_test,
                                r_hat_max=r_hat_max, divergences_nb=div_nb, divergences_perc=div_perc)
        pusher.push_aggregated_orders()
        pusher.push_prior_posterior_mean_sd(pusher.push_id)
        pusher.push_model_configuration(waic=waic, r2_regular_train=r2_regular_train, r2_regular_test=r2_regular_test,
                                        r2_bayesian_train=r2_bayesian_train, r2_bayesian_test=r2_bayesian_test)
        return (pusher.push_id, waic, r2_regular_train, r2_regular_test, r2_bayesian_train, r2_bayesian_test)
