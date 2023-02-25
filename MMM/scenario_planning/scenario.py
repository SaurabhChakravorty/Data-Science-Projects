import collections
from abc import ABC

import config
from db_interaction import pull
from scenario_planning import Channel

"""

To dos:
"""


class Scenario(ABC):

    def __init__(self, push_ids):
        self.__model = None
        self.__set_model(push_ids)
        self.__channels = self.__get_channels()

    @property
    def model(self):
        return self.__model

    @property
    def channels(self):
        return self.__channels

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

    def __get_channels(self):
        channels = self.__instantiate_channels()
        for sm in self.__model.component_container:
            channels[sm.main_effect.channel_name].add_independent(sm.main_effect, sm.main_effect.name + '_in_' + sm.name)
            for ia in sm.interactions:
                channels[ia.channel_name].add_independent(ia, ia.name + '_in_' + sm.name)
        return channels

    def __instantiate_channels(self):
        channels = collections.OrderedDict()
        for sm in self.__model.component_container:
            channel = Channel(sm.main_effect.channel_name)
            channels[sm.main_effect.channel_name] = channel
            for ia in sm.interactions:
                channel = Channel(ia.channel_name)
                channels[ia.channel_name] = channel
        return channels
