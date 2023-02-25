import datetime
import pyodbc
import config

'''
To dos:
* [!] get_scenario_numbers is very ugly: refactor!
* Return dicts by default?
'''


class EntityGateway:

    @staticmethod
    def get_scenario_numbers(run_id):
        file = open('db_interaction/sql/sc_opt_inputs', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        dict = {}
        for value in result:
            _values = []
            if value[2] is None:
                _values.append(0)
            else:
                _values.append(value[2])
            if value[3] is None:
                _values.append(0)
            else:
                _values.append(value[3])
            if value[4] is None:
                _values.append(0)
            else:
                _values.append(value[4])
            if value[5] is None:
                _values.append(0)
            else:
                _values.append(value[5])

            dict[str(value[0])] = (
                str(value[1]),
                float(_values[0]),
                float(_values[1]),
                float(_values[2]),
                float(_values[3]))
        return dict

    @staticmethod
    def get_priors_group_values(run_id):
        file = open('db_interaction/sql/in_priors_group_values', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        return [(str(value[0]), str(value[1]), float(value[2])) for value in result]

    @staticmethod
    def push_opt_steps(opt_steps):
        for step in opt_steps:
            file = open('db_interaction/sql/res_opt_steps', 'r')
            temp = file.read()
            query = temp\
                .replace('@push_id', "'" + str(step[0]) + "'") \
                .replace('@model_name', "'" + str(step[1]) + "'") \
                .replace('@step', "'" + str(step[2]) + "'") \
                .replace('@channel', "'" + str(step[3]) + "'") \
                .replace('@coef', "'" + str(step[4]) + "'") \
                .replace('@value', "'" + str(step[5]) + "'") \
                .replace('@is_improvement', "'" + str(step[6]) + "'") \
                .replace('@is_final', "'" + str(step[7]) + "'") \
                .replace('@waic', "'" + str(step[8]) + "'")
            EntityGateway.__execute_non_query(query)


    @staticmethod
    def push_priors_group(run_id, priors_group):
        value_str = ''
        for _run_id, channel, coef, para, dist, a, b in priors_group.values():
            value_str = value_str + '(' \
                        + str(_run_id) + ',' \
                        + "'" + str(channel) + "'" + ',' \
                        + "'" + str(coef) + "'" + ',' \
                        + "'" + str(para) + "'" + ',' \
                        + "'" + str(dist) + "'" + ',' \
                        + str(a) + ',' \
                        + str(b) \
                        + ')' + ','
        file = open('db_interaction/sql/in_priors_group__push', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id)).replace('@values', value_str[:-1])
        EntityGateway.__execute_non_query(query)

    @staticmethod
    def push_priors_b(run_id, priors_b):
        value_str = ''
        for _run_id, submodel, cpn, cpn_type, dist, a, b in priors_b.values():
            value_str = value_str + '(' \
                        + str(_run_id) + ',' \
                        + "'" + str(submodel) + "'" + ',' \
                        + "'" + str(cpn) + "'" + ',' \
                        + "'" + str(cpn_type) + "'" + ',' \
                        + "'" + str(dist) + "'" + ',' \
                        + str(a) + ',' \
                        + str(b) \
                        + ')' + ','
        file = open('db_interaction/sql/in_priors_b__push', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id)).replace('@values', value_str[:-1])
        EntityGateway.__execute_non_query(query)

    @staticmethod
    def get_interaction_counts(run_id):
        file = open('db_interaction/sql/in_interactions', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        return [(str(value[0]), str(value[1]), float(value[2]), float(value[3])) for value in result]

    @staticmethod
    def get_distinct_list_of_submodels(run_id):
        file = open('db_interaction/sql/in_submodels_distinct', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        return [str(value[0]) for value in result]

    @staticmethod
    def get_channel_values(run_id, channel_name, value_name, date1, date2, include_we):
        if include_we:
            file = open('db_interaction/sql/in_cha_@channel', 'r')
        else:
            file = open('db_interaction/sql/in_cha_@channel__no_we', 'r')
        temp = file.read()
        query = temp \
            .replace('@run_id', str(run_id)) \
            .replace('@channel_name', str(channel_name)) \
            .replace('@value_name', str(value_name)) \
            .replace('@date1', str(date1)) \
            .replace('@date2', str(date2))
        result = EntityGateway.__execute(query)
        if value_name == 'date':
            return [value[0] for value in result]
        else:
            return [float(-999 if value[0] is None else value[0]) for value in result]

    @staticmethod
    def get_dummy_dates(run_id):
        file = open('db_interaction/sql/in_cha_dummy', 'r')
        temp = file.read()
        query = temp \
            .replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        return [value[0] for value in result]

    @staticmethod
    def get_prior_b(run_id, submodel_name, channel_name):
        file = open('db_interaction/sql/in_priors_b', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id)).replace('@submodel_name', str(submodel_name)).replace('@channel_name', str(channel_name))
        result = EntityGateway.__execute(query)
        for value in result:
            return (str(value[0]), float(value[1]), float(value[2]))

    @staticmethod
    def get_sigma(run_id):
        file = open('db_interaction/sql/in_priors_control__sigma', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        for value in result:
            if value[2] is not None:
                b = float(value[2])
            else:
                b = None
            return (str(value[0]), float(value[1]), b)

    @staticmethod
    def get_derived_priors(run_id):
        file = open('db_interaction/sql/in_priors_derived', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        return [(str(value[0]), str(value[1])) for value in result]

    @staticmethod
    def get_group_priors(run_id, channel_name):
        file = open('db_interaction/sql/in_priors_group', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id)).replace('@channel_name', str(channel_name))
        result = EntityGateway.__execute(query)
        out = []
        for value in result:
            if value[5] is not None:
                b = float(value[5])
            else:
                b = None
            out.append((str(value[0]), str(value[1]), str(value[2]), str(value[3]), float(value[4]), b))
        return out  # [i] returns four priors: ela & rate with a & b

    @staticmethod
    def get_days_to_exclude(run_id):
        file = open('db_interaction/sql/in_exclusions', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        return [datetime.datetime.strptime(str(value[0]), '%Y-%m-%d') for value in result]

    @staticmethod
    def get_model_components(run_id, submodel):
        file = open('db_interaction/sql/in_submodels', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id)).replace('@submodel', str(submodel))
        result = EntityGateway.__execute(query)
        return [(str(value[0]), str(value[1])) for value in result]

    @staticmethod
    def get_submodels_and_components(run_id):
        file = open('db_interaction/sql/in_submodels_submodels_components', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        return [(str(value[0]), str(value[1])) for value in result]

    @staticmethod
    def get_distinct_list_of_model_components(run_id):
        file = open('db_interaction/sql/in_submodels_distinct_components', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        return [value[0] for value in result]

    @staticmethod
    def get_control_prior(run_id, control_variable):
        file = open('db_interaction/sql/in_priors_control', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id)).replace('@control_variable', str(control_variable))
        result = EntityGateway.__execute(query)
        for value in result:
            if value[2] is not None:
                b = float(value[2])
            else:
                b = None
            return (str(value[0]), float(value[1]), b)
        return None  # [i] None required since control variables are optional

    @staticmethod
    def get_waic(push_id):
        file = open('db_interaction/sql/res_diagnostics', 'r')
        temp = file.read()
        query = temp.replace('@push_id', str(push_id))
        result = EntityGateway.__execute(query)
        for value in result:
            return value[0]
        return -1  # [i] return -1 when WAIC not available

    @staticmethod
    def get_b_prior_flexibility(run_id, channel):
        file = open('db_interaction/sql/in_priors_conv_coefs_flexibility', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id)).replace('@channel', str(channel))
        result = EntityGateway.__execute(query)
        for value in result:
            return value[0]
        return None

    @staticmethod
    def get_b_prior_flexibility_distinct_channels(run_id):
        file = open('db_interaction/sql/in_priors_conv_coefs_flexibility_distinct_channels', 'r')
        temp = file.read()
        query = temp.replace('@run_id', str(run_id))
        result = EntityGateway.__execute(query)
        return [value[0] for value in result]

    @staticmethod
    def update_history_tables(run_id, push_id, attribution, model_name):
        tables = EntityGateway.__get_all_table_names()

        for table in tables:
            columns = EntityGateway.__get_column_names_of_table(table)
            channels = EntityGateway.get_distinct_list_of_model_components(run_id)
            if (table.startswith('in_') and not table.startswith('in_cha_')) or \
                    (table.startswith('in_cha_') and table.replace('in_cha_', '') in channels) or \
                    (attribution.name == 'tor' and table.startswith('tor_config')):
                if table.startswith('tor_config'):
                    in_table = 'mta.' + table
                    hist_table_short = 'hist_tor_config'
                    hist_table = 'mta.hist_tor_config'
                else:
                    in_table = 'mmm.' + table
                    hist_table_short = table.replace('in_', 'hist_')
                    hist_table = in_table.replace('in_', 'hist_')

                if hist_table_short not in tables:
                    if 'run_id' in columns:
                        file = open('db_interaction/sql/hist__create_new_table_with_run_id', 'r')
                        temp = file.read()
                        query = temp.replace('@run_id', str(run_id)).replace('@push_id', str(push_id)).replace('@model_name',
                            model_name).replace('@in_table', str(in_table)).replace('@hist_table', str(hist_table))
                    else:
                        file = open('db_interaction/sql/hist__create_new_table_without_run_id', 'r')
                        temp = file.read()
                        query = temp.replace('@push_id', str(push_id)).replace('@model_name', model_name).replace(
                            '@in_table', str(in_table)).replace('@hist_table', str(hist_table))
                    EntityGateway.__execute_non_query(query)
                else:
                    if 'run_id' in columns:
                        file = open('db_interaction/sql/hist__update_table_with_run_id', 'r')
                        temp = file.read()
                        query = temp.replace('@run_id', str(run_id)).replace('@push_id', str(push_id)).replace('@model_name',
                             model_name).replace('@in_table', str(in_table)).replace('@hist_table', str(hist_table))
                    else:
                        file = open('db_interaction/sql/hist__update_table_without_run_id', 'r')
                        temp = file.read()
                        query = temp.replace('@push_id', str(push_id)).replace('@model_name', model_name).replace(
                            '@in_table', str(in_table)).replace('@hist_table', str(hist_table))
                    EntityGateway.__execute_non_query(query)
        print('[i] added snapshot of in_ tables to hist_ tables')
        return None

    @staticmethod
    def delete_history_tables_entry(model_name):
        tables = EntityGateway.__get_all_table_names()
        for table in tables:
            if table.startswith('hist_'):
                if 'tor_config' in table:
                    hist_table = 'mta.' + table
                else:
                    hist_table = 'mmm.' + table
                file = open('db_interaction/sql/hist__remove_model_entry', 'r')
                temp = file.read()
                query = temp.replace('@model_name', model_name).replace('@hist_table', hist_table)
                EntityGateway.__execute_non_query(query)
        print('[i] removed history table entries with model_name = ' + model_name)

    @staticmethod
    def __get_all_table_names():
        cnx = pyodbc.connect(config.Config.db_connection_string)
        cursor = cnx.cursor()
        tables = []
        for row in cursor.tables():
            tables.append(row.table_name)
        return tables

    @staticmethod
    def __get_column_names_of_table(table):
        cnx = pyodbc.connect(config.Config.db_connection_string)
        cursor = cnx.cursor()
        columns = []
        for row in cursor.columns(table=table):
            columns.append(row.column_name)
        return columns

    @staticmethod
    def get_distinct_list_of_model_names():
        file = open('db_interaction/sql/res_diagnostics_get_distinct_model_names', 'r')
        query = file.read()
        result = EntityGateway.__execute(query)
        return [value[0] for value in result]

    @staticmethod
    def __execute(query):
        cnx = pyodbc.connect(config.Config.db_connection_string)
        cursor = cnx.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        cnx.close()
        return result

    @staticmethod
    def __execute_non_query(query):
        cnx = pyodbc.connect(config.Config.db_connection_string)
        cursor = cnx.cursor()
        cursor.execute(query)
        cnx.commit()
        cursor.close()
        cnx.close()
