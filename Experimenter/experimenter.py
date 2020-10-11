from dataclasses import dataclass

from utils import DotDict

from Games.hex.HexGame import HexGame
from Games.othello.OthelloGame import OthelloGame
from Games.gym.GymGame import GymGame
from Games.atari.AtariGame import AtariGame

from Experimenter.Players import *
from Games.hex.MuZeroModel.NNet import NNetWrapper as HexMuZero
from Games.hex.AlphaZeroModel.NNet import NNetWrapper as HexAlphaZero
from Games.othello.AlphaZeroModel.NNet import NNetWrapper as OthelloAlphaZero

from AlphaZero.MCTS import MCTS
from MuZero.MuMCTS import MuZeroMCTS


@dataclass
class PlayerConfig:
    name: str

    def __init__(self, name: str, player_cls) -> None :
        self.name = name
        self.player = player_cls


@dataclass
class GameConfig:
    name: str

    def __init__(self, name: str, cls, alpha_zero, mu_zero) -> None:
        self.name = name
        self.cls = cls
        self.alpha_zero = alpha_zero
        self.mu_zero = mu_zero


class ExperimentConfig(object):

    players = DotDict({
        "ALPHAZERO": PlayerConfig("ALPHAZERO", AlphaZeroPlayer),
        "MUZERO": PlayerConfig("MUZERO", MuZeroPlayer),
        "RANDOM": PlayerConfig("RANDOM", RandomPlayer),
        "DETERMINISTIC": PlayerConfig("DETERMINISTIC", DeterministicPlayer),
        "MANUAL": PlayerConfig("MANUAL", ManualPlayer)
    })

    games = DotDict({
        "HEX": GameConfig("HEX", HexGame, HexAlphaZero, HexMuZero),
        "OTHELLO": GameConfig("OTHELLO", OthelloGame, OthelloAlphaZero, None),
        "GYM": GameConfig("GYM", GymGame, None, None),
        "ATARI": GameConfig("GYM", AtariGame, None, None)
    })

    def __init__(self, experiment_file: str):
        self.experiment_file = experiment_file
        self.game = None
        self.player_configs = list()

    def construct(self) -> None:
        experiment = DotDict.from_json(self.experiment_file)

        env = experiment.environment

        if env.name in self.games:
            self.game = self.games[env.name]
        else:
            raise NotImplementedError("Did not specify a valid environment.")

        g = self.game.cls(**env.args)

        player_configs = experiment.players
        for config in player_configs:

            if config.name not in self.players:
                raise NotImplementedError("Did not specify a valid environment.")

            if config.name == self.players.ALPHAZERO.name:
                algorithm_config = DotDict.from_json(config.config)

                model = self.game.alpha_zero(g, algorithm_config.net_args)
                search = MCTS(g, model, algorithm_config.args)
                self.players.ALPHAZERO.player(g, search, model)

                self.player_configs.append(self.players.ALPHAZERO.player(g, search, model))

            elif config.name == self.players.MUZERO.name:
                algorithm_config = DotDict.from_json(config.config)

                model = self.game.mu_zero(g, algorithm_config.net_args)
                search = MuZeroMCTS(g, model, algorithm_config.args)
                self.players.MUZERO.player(g, search, model)

                self.player_configs.append(self.players.MUZERO.player(g, search, model))

            else:
                self.player_configs.append(self.players[config.name].player(g))


def tournament_final(experiment: ExperimentConfig) -> None:
    pass


def tournament_pool(experiment: ExperimentConfig) -> None:
    pass
