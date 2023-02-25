import math

import matplotlib.pyplot as plt
from numpy import power
from scipy.optimize import curve_fit

from db_interaction.entity_gateway import EntityGateway
from evaluation.attribution import Attribution
from scenario_planning.channel import Investment_Type
from scenario_planning.scenario import Scenario

"""

To dos:
"""


class SpendScenario(Scenario):

    def __init__(self, run_id, push_ids):
        super().__init__(push_ids)
        self.__set_investment_props(run_id)

    @property
    def orders_total(self):
        results = Attribution.get_aggregated_orders_per_channel(self.model.component_container)
        return sum(results.values())

    def run(self, spend_new):
        for channel in self.channels.values():
            channel.change_spend(spend_new[channel.name])

    def __set_investment_props(self, run_id):
        scenario = EntityGateway.get_scenario_numbers(run_id)
        for channel in self.channels.values():
            channel.inv_type = Investment_Type[scenario[channel.name][0]]
            channel.spend_base = scenario[channel.name][1]
            channel.cpa_base = scenario[channel.name][4]
            channel.min = scenario[channel.name][2]
            channel.max = scenario[channel.name][3]

    # region response_curves

    def plot_response_curves_all(self):
        for key, channel in self.channels.items():
            if channel.cpa_base is not None and channel.cpa_base != 0:
                self.plot_response_curves_single(channel.name, show_plot=False)
        plt.show()

    def plot_response_curves_single(self, channel_name, show_plot=True):
        if channel_name not in self.channels:
            raise ValueError("Channel not found.")
        # generate spends and orders
        channel = self.channels[channel_name]
        spends = [i * .1 * channel.spend_base for i in range(1, 30)]
        orders = self.__get_orders(channel, spends)
        self.__remove_nans(spends, orders)
        if len(orders) <= 2:
            print('Cannot plot ' + channel.name + ' because orders could not be determined.')
            return
        # fit and plot
        popt, pcov = curve_fit(self.__f_orders, spends, orders)
        if popt[0] == 1. and popt[1] == 1.:
            print('Cannot plot ' + channel.name + ' because covariance of the parameters could not be estimated.')
            return
        self.__plot_response_curves(spends, channel.spend_base, popt, channel.name)
        if show_plot:
            plt.show()

    def __get_orders(self, channel, spends):
        orders = []
        for spend in spends:
            channel.change_spend(spend)
            att = Attribution.get_aggregated_orders_per_channel(self.model.component_container)
            orders.append(float(att[channel.name]))
        return orders

    def __remove_nans(self, spends, orders):
        ipop = [i for i, o in enumerate(orders) if math.isnan(o)]
        for i in sorted(ipop, reverse=True):
            orders.pop(i)
            spends.pop(i)

    def __f_orders(self, x, b, z):
        return b * power(x, z)

    def __f_mc(self, o, b, z):
        return power(o, 1 / z - 1) / (power(b, 1 / z) * z)

    def __plot_response_curves(self, spends, spend_base, popt, title='Response curves'):
        orders = self.__f_orders(spends, *popt)
        mcs = self.__f_mc(orders, *popt)
        orders_base = self.__f_orders(spend_base, *popt)
        mc_base = self.__f_mc(orders_base, *popt)
        if mc_base > 1000000:
            mc_base_label = '> 1M'
        else:
            mc_base_label = '= ' + str(round(mc_base, 2))
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.title.set_text(title)
        ax1.plot(orders, spends, label='Orders', color='b')
        ax1.annotate("    Current spend = " + str(round(spend_base, 2)) + "\n" +
                     "    Attributed orders = " + str(round(orders_base, 0)), (orders_base, spend_base),
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='grey', boxstyle='round,pad=0.5'))
        ax1.plot(orders_base, spend_base, "ro", color='k')
        ax2.plot(orders, mcs, label='Marginal cost', color='g')
        ax2.annotate("    Current marginal cost " + mc_base_label, (orders_base, mc_base),
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='grey', boxstyle='round,pad=0.5'))
        ax2.plot(orders_base, mc_base, "ro", color='k')
        ax1.set_xlabel('Orders')
        ax1.set_ylabel('Spend')
        ax2.set_ylabel('Marginal cost')
        ax1.legend(loc="lower right")
        ax2.legend(loc="upper right")
        plt.legend()

    # endregion
