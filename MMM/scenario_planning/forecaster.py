from scenario_planning.value_change_scenario import ValueChangeScenario

'''
To do's:
- Add forward forecasting functionality (with different independent assumptions?)

Notes ae as of 2020-11-10:
* set_forecast: Separate model reference not needed --> refactored [now killed]
* set_forecast: sm.get_result_values() doesn't do anything --> killed
* oberserved: 0 not allowed in fitting --> kept consistent [still necessary?]
* independent: Overload override_values method --> refactored slightly as independent should not know about forecaster
* submodel: 'get_dates' can be shortened --> refactored
* general: Use setter or 'overwrite' instead of 'override'? --> changed to 'overwrite'
* Forecaster should be obsolete --> kept it alive, use as wrapper?
'''


class Forecaster:

    def __init__(self, run_id, push_ids, dates, include_we):
        self.__scenario = ValueChangeScenario(push_ids)
        self.__scenario.run(run_id, dates, include_we)

    @property
    def model(self):
        return self.__scenario.model
