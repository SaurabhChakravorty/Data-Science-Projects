import sys
import config
from fitting.fitter import Fitter
from modeling.enums import Observed_Type
from prior_factory.log_logit import Attribution_Type, LogLogit
from prior_factory.distribution_provider import DistributionProvider
from prior_factory.distribution_provider import Distribution_Type
from prior_factory.prior_predictive_sampler import PPS_LogLogit
from db_interaction.pull import Puller
from evaluation.result_reporter import ResultReporter
from evaluation.attribution import Attribution
from db_interaction.entity_gateway import EntityGateway
from scenario_planning.spend_scenario import SpendScenario





if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 1:
        args = (None, 800, 400, 2, 1)
    else:
        args = sys.argv


    '''
    from scenario_planning.optimizer import Optimizer
    # opt = Optimizer(run_id=22, push_ids=['20210309080418'])  ## Foundation
    # opt.maximize_orders(spend=287370.17)
    opt = Optimizer(run_id=15, push_ids=['20221117102007'])  ## DA
    opt.maximize_orders(spend= 86706.00 , initial_changes=10000)  # 
    '''

    '''
    ssc = SpendScenario(run_id=33, push_ids=['20210712075042'])
    ssc.plot_response_curves_all()
    '''


    # set model settings
    model_name = '22Q1_22Q4_QF_SampleTech_3'
    observed_type = Observed_Type.lc_orders  # lc_revenue
    run_id = 25
    sd = 0.2
    dates = ('2022-01-01', '2022-12-31') #  ('2020-01-01', '2020-12-31')  # ('2020-01-01', '2020-03-31')
    test_dates = ('2022-01-01','2022-02-01','2022-03-01','2022-04-01','2022-05-01','2022-06-01','2022-07-01')  # None
    attribution = Attribution_Type.tor
    

    # fit posterior model

    fitter = Fitter()
    fitter.draws, fitter.tune, fitter.chains, fitter.cores = tuple(map(int, args[1:]))
    info = fitter.fit_and_push_log_const(model_name=model_name, run_id=run_id, sd=sd, dates=dates, observed_type=observed_type, attribution_type=attribution, test_dates=test_dates, fit_all_coefs=True)
    attribution_type=attribution
    info = fitter.fit_and_push_log_logit(model_name=model_name, run_id=run_id, sd=sd, dates=dates, observed_type=observed_type, attribution_type=attribution, test_dates=test_dates)
    print('push id: ' + str(info[0]))
    print('waic   : ' + str(info[1]))
    print('r2_regular_train    : ' + str(info[2]))
    print('r2_regular_test    : ' + str(info[3]))
    print('r2_bayesian_train    : ' + str(info[4]))
    print('r2_bayesian_test    : ' + str(info[5]))
    EntityGateway.delete_history_tables_entry(model_name)


    '''
    # sample prior model
    pps = PPS_LogLogit.instance().get_prior_model(run_id=run_id, sd=sd, dates=dates, attribution=attribution, observed_type=observed_type)
    reporter = ResultReporter(pps)
    reporter.show_pred_vs_obs_over_time(save_to_xlsx=True)
    '''

    '''
    # run value change scenario (i.e. plot performance on test set
    from scenario_planning.value_change_scenario import ValueChangeScenario
    push_ids = [info[0]]
    vcs = ValueChangeScenario(push_ids=push_ids)
    run_id = 7
    dates = ('2019-04-01', '2021-06-30')
    vcs.run(run_id=run_id, dates=dates, include_we=False)
    rr = ResultReporter(vcs.model, push_ids)
    rr.show_pred_vs_obs_over_time(save_to_xlsx=False)
    '''

    '''
    #pull fitted model
    push_id = '20230216234037'  # info[0]
    puller = Puller(con=config.Config.db_connection_string)
    model = puller.pull_model(push_id)
    # reporter
    reporter = ResultReporter(model, [push_id])
    # reporter.show_prior_posterior_plot_custom()
    reporter.show_pred_vs_obs_over_time(save_to_xlsx=True)
    #reporter.print_stat_summary()
    #reporter.show_trace_plot()
    #reporter.show_heatmap_plot()
    #reporter.show_posterior_pred_plot()
    #reporter.show_correlogram_plot()
    #att = Attribution.get_aggregated_orders_per_channel(model.component_container)
    #print(att)
    #reporter.show_prior_posterior_plot(plot_varnames=['main_paid_search_brand_b'])
    #reporter.show_prior_posterior_plot_custom(n=4)
    #reporter.show_divergent_transitions_plot(n=12)
    #print(reporter.get_divergences())
    '''
    '''
    from scenario_planning.optimizer import Optimizer
    opt = Optimizer(run_id=25, push_ids=['20230216234037'])  ## ST
    opt.maximize_orders(spend=  400000.00 , initial_changes=1000)
    '''