import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

import config
from db_interaction.entity_gateway import EntityGateway
from db_interaction.pull import Puller
from modeling import MainEffect, Observed, Submodel, QIAmmm
from modeling.enums import Observed_Type


class PredictorEvaluator:

    def __init__(self, run_id, dates, include_we, observed_type=Observed_Type.lc_orders):
        if not isinstance(observed_type, Observed_Type):
            raise TypeError('dependent_type must be an instance of Dependent_Type Enum')
        self.__run_id = run_id
        self.__dates = dates
        self.__include_we = include_we

    def __get_channel_df(self, channel_name, raw):
        if raw is True:
            puller = Puller(con=config.Config.db_connection_string)
            df = puller.pull_input_data(self.__run_id, channel_name=channel_name)
            return df
        else:
            lc_orders = Observed(self.__run_id, channel_name, self.__dates, self.__include_we).values
            value = MainEffect(self.__run_id, channel_name, self.__dates, self.__include_we, set_prior_b=False).values
            date = EntityGateway.get_channel_values(self.__run_id, channel_name, 'date',
                                                    self.__dates[0], self.__dates[1], self.__include_we)
            return pd.DataFrame({'date': date, 'lc_orders': lc_orders, 'value': value})

    def __get_all_model_inputs(self):
        model = QIAmmm(self.__run_id)
        submodel_names = EntityGateway.get_distinct_list_of_submodels(self.__run_id)
        for name in submodel_names:
            model.add_submodel(Submodel(self.__run_id, name, self.__dates, self.__include_we,
                                        observed_type=Observed_Type.lc_orders))
        dfs = {}
        for key, value in model._QIAmmm__submodels.items():
            '''
            # The name of the key is a tuple. However, I can't manage to replace 'paid_search' with 'sea' in all tuples:
            key_name = key.replace('_lc_orders', '')
            key_name = tuple([sub.replace('paid_search', 'sea') for sub in key_name])
            dfs[key_name] = value._Submodel__df
            '''
            dfs[key.replace('_lc_orders', '').replace('paid_search', 'sea')] = value._Submodel__df
        df = pd.concat(dfs, axis=1, join='outer')
        df.columns = df.columns.to_flat_index()
        return df

    def show_predictor_vs_observed(self, channel_name, raw=False):
        df = self.__get_channel_df(channel_name, raw)
        df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d')
        corr = df[['value', 'lc_orders']].corr().iloc[0, 1]
        label = 'r = ' + str("%.3f" % corr)
        xpos = 0.80 * max(df['value'])
        ypos = 0.95 * max(df['lc_orders'])
        ax = df.plot.scatter(x='value',
                             y='lc_orders',
                             c=df['date'].dt.dayofyear,
                             colormap='viridis',
                             title=channel_name).annotate(label, (xpos, ypos))
        plt.savefig('show_predictor_observed_with_time' + '.png')

    def show_predictor_observed_over_time(self, channel_name, raw=False):
        df = self.__get_channel_df(channel_name, raw)
        df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d')
        df['lc_orders'].plot(legend=True)
        df['value'].plot(secondary_y=True, legend=True, title=channel_name)

    def show_heatmap(self):
        df = self.__get_all_model_inputs()
        corrMatrix = df.corr()
        plt.figure(figsize=(40, 20))
        sn.set(font_scale=0.7)
        chart = sn.heatmap(corrMatrix, annot=True)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=6)
        chart.set_yticklabels(chart.get_yticklabels(), rotation=0, fontsize=8)
        chart.set(xlabel=None, ylabel=None)
        plt.savefig('show_heatmap_plot' + '.png')

    def show_corr_model_inputs(self, x, y):
        df = self.__get_all_model_inputs()
        corr = df[[x, y]].corr().iloc[0, 1]
        label = 'r = ' + str("%.3f" % corr)
        xpos = 0.80 * max(df[x])
        ypos = 0.95 * max(df[y])
        ax = df.plot.scatter(x=x, y=y).annotate(label, (xpos, ypos))
