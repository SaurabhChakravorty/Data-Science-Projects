from db_interaction.entity_gateway import EntityGateway
from scenario_planning.scenario import Scenario

"""

To dos:
"""


class ValueChangeScenario(Scenario):

    def __init__(self, push_ids):
        super().__init__(push_ids)

    def run(self, run_id, dates, include_we):
        for channel in self.channels.values():
            values = EntityGateway.get_channel_values(run_id, channel.name, 'value', dates[0], dates[1], include_we)
            channel.change_values(values)
        for sm in self.model.component_container:
            sm.overwrite_dates(run_id=run_id, dates=dates, include_we=include_we)
            values = EntityGateway.get_channel_values(run_id, sm.dependent.submodel_name, 'lc_orders', dates[0], dates[1], include_we)
            sm.dependent.overwrite_values(values)
        self.model.close()
