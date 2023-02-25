from scipy.optimize import fmin_cobyla

from scenario_planning.spend_scenario import SpendScenario

"""
cobyla reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cobyla.html

To dos:
"""


class Optimizer:

    def __init__(self, run_id, push_ids):
        self.__scenario = SpendScenario(run_id, push_ids)
        self.__max_spend = -1
        self.__initial_guess = [channel.spend_base for channel in self.__scenario.channels.values()]
        self.__scenario.run(self.__get_input_dict(self.__initial_guess))
        self.__counter = 0

    @property
    def scenario_manager(self):
        return self.__scenario

    def maximize_orders(self, spend, initial_changes=5000, final_accuracy=1e-7, max_iterations=10000):
        self.__counter = 0
        self.__write_iteration_step('ini', self.__initial_guess)
        self.__max_spend = spend
        results = fmin_cobyla(func=self.__objective_max, x0=self.__initial_guess,
                              cons=[self.__con_not_negative, self.__con_not_greater_than, self.__con_spend_limit, self.__con_not_smaller_than],
                              rhobeg=initial_changes, rhoend=final_accuracy, maxfun=max_iterations)
        self.__scenario.run(self.__get_input_dict(results))
        self.__write_iteration_step('final', results)

    def __write_iteration_step(self, header, x):
        self.__counter += 1
        total = round(self.__scenario.orders_total, 1)
        dict = self.__get_input_dict(x)
        spends = '[' + str(self.__counter) + '] ' + header + ' -> total: ' + str(total) + ' '
        for key in dict:
            spends += key + ': ' + str(round(dict[key], 1)) + ' '
        print(spends)

    def __objective_max(self, x):
        input_dict = self.__get_input_dict(x)
        self.__scenario.run(input_dict)
        self.__write_iteration_step('temp', x)
        return -1 * self.__scenario.orders_total

    def __con_not_negative(self, x):
        return x

    def __con_not_greater_than(self, x):
        input_dict = self.__get_input_dict(x)
        return [self.__scenario.channels[key].max - input_dict[key] for key in input_dict]

    def __con_not_smaller_than(self, x):
        input_dict = self.__get_input_dict(x)
        return [input_dict[key] - self.__scenario.channels[key].min for key in input_dict]

    def __con_spend_limit(self, x):
        return self.__max_spend - sum(x)

    def __get_input_dict(self, x):
        dict = {}
        for i, channel in enumerate(self.__scenario.channels.keys(), start=0):
            dict[channel] = x[i]
        return dict
