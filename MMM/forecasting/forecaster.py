from db_interaction.entity_gateway import EntityGateway
from modeling.independent import Independent
from modeling.observed import Observed



class Forecaster:
    def __init__(self, push_ids, dates):
        self.__model = None
        self.__set_model(push_ids)
        self.__channels = self.__get_channels()
        self.__dates = dates

    # region properties

    @property
    def model(self):
        return self.__model

    @property
    def channels(self):
        return self.__channels

'''
could try that later
    def __set_model(self, push_ids):
        puller = pull.Puller(con=config.Config.db_connection_string)
        for push_id in push_ids:
            model = puller.pull_model(push_id)
            if self.__model is None:
                self.__model = model
            else:
                for sm in model.component_container:
                    self.__model.add_submodel(sm)
        self.__model.close()
'''
    def __set_variables(self, run_id, dates, include_we):
        for sm in self.__model.component_container:
            forecast_observed = EntityGateway.get_channel_values(run_id, sm.name, 'lc_orders', dates[0], dates[1], include_we)
            sm.dependent.values.replace_observed(forecast_observed)
            for ia in sm.interactions:
                forecast_independent = EntityGateway.get_channel_values(run_id, ia.name, 'value', dates[0], dates[1], include_we)
                scale_factor = 1
                ia.values.override_values(forecast_independent, scale_factor)
