from enum import Enum
import typing

from utils import DotDict


class Players(Enum):
    ALPHAZERO: int = 1
    MUZERO: int = 2
    RANDOM: int = 3
    GREEDY: int = 4


class Games(Enum):
    HEX: int = 1
    OTHELLO: int = 2
    GYM: int = 3
    ATARI: int = 4


class Experimenter(object):

    def __init__(self, environment_config: DotDict, player_configurations: typing.List[DotDict, ...]):
        self.environment_config = environment_config
        self.player_configs = player_configurations
        self.players = list()

    def add_player(self, configuration):
        self.player_configs.append(configuration)

    def construct(self):
        for config in self.player_configs:
            p = Players[config.algorithm]

            if p == Players.ALPHAZERO:
                pass
            elif p == Players.MUZERO:
                pass
            elif p == Players.RANDOM:
                pass
            elif p == Players.GREEDY:
                pass
            else:
                raise NotImplementedError("Did not specify a valid algorithm.")







