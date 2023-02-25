from db_interaction.entity_gateway import EntityGateway
from modeling.submodel import Submodel

import pyodbc


class InputScanner:
    def __init__(self, run_id, observed_type, dates, test_dates, include_we):
        self.__run_id = run_id
        self.__observed_type = observed_type
        self.__dates = dates
        self.__test_dates = test_dates
        if self.__test_dates is not None:
            self.__date1, self.__date2 = min(self.__dates + self.__test_dates), max(self.__dates + self.__test_dates)
        else:
            self.__date1, self.__date2 = self.__dates[0], self.__dates[1]
        self.__include_we = include_we

    def run_input_scanner(self):
        exceptions = str()
        if self.__compare_submodels_interactions() is not None:
            exceptions += self.__compare_submodels_interactions()
        if self.__compare_submodels_flexibility() is not None:
            exceptions += self.__compare_submodels_flexibility()
        if self.__compare_submodels_prior_group_values() is not None:
            exceptions += self.__compare_submodels_prior_group_values()
        if self.__check_submodels_main_effects() is not None:
            exceptions += self.__check_submodels_main_effects()
        if self.__check_missing_tables_in_cha() is not None:
            exceptions += self.__check_missing_tables_in_cha()
        if self.__check_observed_in_cha() is not None:
            exceptions += self.__check_observed_in_cha()
        if self.__check_dates_in_cha() is not None:
            exceptions += self.__check_dates_in_cha()
        if len(exceptions) > 0:
            raise Exception("[!] Input scanner detected missing or incompatible data in the DB that prevents the model from running:\n" + exceptions)

    def __compare_submodels_interactions(self):
        ia_counts = EntityGateway.get_interaction_counts(self.__run_id)
        ias = [ia[0:2] for ia in ia_counts]
        cpns = EntityGateway.get_submodels_and_components(self.__run_id)
        missing_ias = set(cpns) - set(ias)
        if len(missing_ias) > 0:
            return "--> Last_contact - interactor combination missing in mmm.in_interactions: " + str(missing_ias) + "\n"

    def __compare_submodels_flexibility(self):
        components = EntityGateway.get_distinct_list_of_model_components(self.__run_id)
        flexibilities = EntityGateway.get_b_prior_flexibility_distinct_channels(self.__run_id)
        missing_flexs = set(components) - set(flexibilities)
        if len(missing_flexs) > 0:
            return "--> Channel missing in mmm.in_prior_conv_coefs_flexibility: " + str(missing_flexs) + "\n"

    def __compare_submodels_prior_group_values(self):
        components = EntityGateway.get_distinct_list_of_model_components(self.__run_id)
        prior_values = EntityGateway.get_priors_group_values(self.__run_id)
        prior_channels = [channel[0] for channel in prior_values]
        missing_priors = set(components) - set(prior_channels)
        if len(missing_priors) > 0:
            return "--> Channel missing in mmm.in_priors_group_values: " + str(missing_priors) + "\n"

    def __check_submodels_main_effects(self):
        submodels = EntityGateway.get_distinct_list_of_submodels(self.__run_id)
        cpns = EntityGateway.get_submodels_and_components(self.__run_id)
        cpns_main = [cpn[0] for cpn in cpns if cpn[0] == cpn[1]]
        missing_main_effects = set(submodels) - set(cpns_main)
        if len(missing_main_effects) > 0:
            return "--> Main effect missing or False while interaction(s) are True for following submodel(s) in mmm.in_submodels: " + str(missing_main_effects) + "\n"

    def __check_missing_tables_in_cha(self):
        components = EntityGateway.get_distinct_list_of_model_components(self.__run_id)
        missing_tables = []
        for cpn in components:
            table_exists = self.__check_table_exists(cpn)
            if not table_exists:
                missing_tables.append(cpn)
        if len(missing_tables) > 0:
            return "--> Missing mmm.in_cha table(s): " + ', '.join([str(ch) for ch in missing_tables]) + "\n"

    def __check_observed_in_cha(self):
        submodels = EntityGateway.get_distinct_list_of_submodels(self.__run_id)
        incomplete_channels = []
        for sm in submodels:
            table_exists = self.__check_table_exists(sm)
            if table_exists:
                observed = EntityGateway.get_channel_values(self.__run_id, sm, str(self.__observed_type.value),
                                                            self.__date1, self.__date2,  self.__include_we)
                if -999 in observed:
                    incomplete_channels.append(sm)
        if len(incomplete_channels) > 0:
            return "--> " + str(self.__observed_type.value) + " includes -999 or None in mmm.in_cha table(s): " + \
                   ', '.join([str(ch) for ch in incomplete_channels]) + "\n"

    def __check_dates_in_cha(self):
        date_list = Submodel.get_date_list(self.__dates, self.__run_id, self.__include_we)
        date_list = [date.strftime("%Y-%m-%d") for date in date_list]
        components = EntityGateway.get_distinct_list_of_model_components(self.__run_id)
        missing_dates = {}
        for cpn in components:
            table_exists = self.__check_table_exists(cpn)
            if table_exists:
                cpn_dates = EntityGateway.get_channel_values(self.__run_id, cpn, 'date',
                                                             self.__date1, self.__date2,  self.__include_we)
                cpn_dates = [date.strftime("%Y-%m-%d") for date in cpn_dates]
                compare_dates = set(date_list) - set(cpn_dates)
                if len(compare_dates) > 0:
                    missing_dates[cpn] = (min(compare_dates), max(compare_dates))
        if len(missing_dates) > 0:
            return "--> Missing dates in mmm.in_cha_ table(s): " + \
                   str(missing_dates) + "\n"

    def __check_table_exists(self, channel):
        try:
            EntityGateway.get_channel_values(self.__run_id, channel, 'date',
                                             self.__date1, self.__date2, self.__include_we)
        except pyodbc.ProgrammingError:
            return False
        return True
