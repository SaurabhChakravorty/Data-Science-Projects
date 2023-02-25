from modeling.enums import Control_Type

'''
To dos:
'''


class Attribution:

    @staticmethod
    def get_aggregated_orders_per_channel(component_container):
        att = Attribution.get_attribution_per_date_and_channel(component_container)
        dict = {}
        for ch, values in att.items():
            if ch in dict:
                dict[ch] += sum(values)
            else:
                dict[ch] = sum(values)
        return dict

    @staticmethod
    def get_attribution_per_date_and_channel(component_container):
        '''
        Returns attributed orders per channel per day.
        '''
        att = Attribution.get_attribution_per_date_and_submodel_and_channel(component_container)
        dict = {}
        for sm, chs in att.items():
            for ch, values in chs.items():
                if ch in dict:
                    dict[ch] = [a + b for a, b in zip(dict[ch], values)]
                else:
                    dict[ch] = values
        return dict

    @staticmethod
    def get_attribution_per_date_and_submodel_and_channel(component_container):
        '''
        Returns attributed orders per submodel and channel per day.
        Interactions are attributed 50/50 to the participating channels.
        '''
        att = {}  # sm (dict) | channel (dict) | attribution values per day (list)
        for sm in component_container:
            att[sm.name] = {}
            att[sm.name][sm.main_effect.channel_name] = sm.main_effect.result_values
            for ia in sm.interactions:
                if sm.name == 'general_demand':  # general demand --> all ia orders go to ia
                    att[sm.name][ia.channel_name] = [v for v in ia.result_values]
                else:
                    att[sm.name][sm.main_effect.channel_name] = [v_ia / 2 + v_main for v_main, v_ia in zip(att[sm.name][sm.main_effect.channel_name], ia.result_values)]
                    att[sm.name][ia.channel_name] = [v / 2 for v in ia.result_values]
            att = Attribution.__add_control_variable_results(att, sm)
        return att

    @staticmethod
    def __add_control_variable_results(att, sm):
        '''
        Control variables are fully attributed to the channels based on their daily share of the orders.
        '''
        # get pre-calculated dummy dict to speed up process
        dummy_result_values = {}
        for d in sm.dummies:
            dummy_result_values[d.name] = sm.get_dummy_result_values(d)

        att_totals_noncontrol = [sum(x) for x in zip(*att[sm.name].values())]
        for channel in att[sm.name].keys():  # iterate over each channel in submodel
            for i, v in enumerate(att[sm.name][channel]):  # iterate over each day
                share = v / att_totals_noncontrol[i]
                if Control_Type.intercept.value in sm.control_priors:
                    att[sm.name][channel][i] += share * sm.intercept_result_values[i]
                if Control_Type.trend.value in sm.control_priors:
                    att[sm.name][channel][i] += share * sm.trend_result_values[i]
                if Control_Type.we.value in sm.control_priors:
                    att[sm.name][channel][i] += share * sm.weekend_result_values[i]
                for d in sm.dummies:
                    att[sm.name][channel][i] += share * dummy_result_values[d.name][i]
        return att
