import bz2
import pickle

import pandas as pd

import db_interaction.db_helper as helper

"""
To dos:
* Should be merged with entity_gateway
* Data types of results should be made explicit
"""


class Puller:

    def __init__(self, con):
        self.__con = con

    def pull_model(self, push_id):
        cnx = helper.get_cnx()
        cursor = cnx.cursor()
        q_pull = 'select top 1 object from mmm.res_model_objects where push_id=' + str(push_id)
        cursor.execute(q_pull)
        model_compr = bz2.decompress(cursor.fetchone()[0])
        model = pickle.loads(model_compr)
        cursor.close()
        cnx.close()
        return model

    def pull_input_data(self, run_id, channel_name):
        cnx = helper.get_cnx()
        q_pull = 'select * from mmm.in_cha_' + channel_name + ' where run_id=' + str(run_id)
        data = pd.read_sql(q_pull, cnx)
        return data

    def pull_posterior_mean_sd(self, push_id):
        cnx = helper.get_cnx()
        q_pull = 'select push_id, model_name, coefficient, mean, sd from mmm.res_sampling where push_id=' + str(push_id)
        data = pd.read_sql(q_pull, cnx)
        return data

    def pull_prior_derived_distribution(self, run_id, coef):
        cnx = helper.get_cnx()
        q_pull = "select distribution from mmm.in_priors_derived where run_id=" + str(run_id) + \
                 " and coef='" + coef + "'"
        value = pd.read_sql(q_pull, cnx).iloc[0, 0]
        return value

    def pull_group_prior_value(self, run_id, channel, coef):
        cnx = helper.get_cnx()
        q_pull = "select value from mmm.in_priors_group_values where run_id=" + str(run_id) + \
                 " and channel='" + channel + "' and coef='" + coef + "'"
        value = pd.read_sql(q_pull, cnx).iloc[0, 0]
        return value
