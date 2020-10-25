from dataclasses import dataclass
from itertools import combinations
import sys

from tqdm import tqdm, trange

from utils import DotDict

from Arena import Arena

from Games.hex.HexGame import HexGame
from Games.othello.OthelloGame import OthelloGame
from Games.gym.GymGame import GymGame
from Games.atari.AtariGame import AtariGame

from Experimenter.players import *
from MuZero.models.DefaultMuZero import MuZeroDefault as HexMuZero  # TODO
from Games.hex.AlphaZeroModel.NNet import NNetWrapper as HexAlphaZero
from Games.othello.AlphaZeroModel.NNet import NNetWrapper as OthelloAlphaZero

from AlphaZero.AlphaMCTS import MCTS
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
        self.experiment_args = DotDict.from_json(experiment_file)
        self.game_config = None
        self.game = None
        self.player_configs = list()

    def construct(self) -> None:
        env = self.experiment_args.environment

        if env.name in self.games:
            self.game_config = self.games[env.name]
        else:
            raise NotImplementedError("Did not specify a valid environment.")

        self.game = self.game_config.cls(**env.args)

        player_configs = self.experiment_args.players
        for config in player_configs:

            if config.name not in self.players:
                raise NotImplementedError("Did not specify a valid environment.")

            if config.name == self.players.ALPHAZERO.name:
                algorithm_config = DotDict.from_json(config.config)

                model = self.game_config.alpha_zero(self.game, algorithm_config.net_args)
                search = MCTS(self.game, model, algorithm_config.args)

                self.player_configs.append(self.players.ALPHAZERO.player(
                    self.game, search, model, config=algorithm_config))

            elif config.name == self.players.MUZERO.name:
                algorithm_config = DotDict.from_json(config.config)

                model = self.game_config.mu_zero(self.game, algorithm_config.net_args)
                search = MuZeroMCTS(self.game, model, algorithm_config.args)

                self.player_configs.append(self.players.MUZERO.player(
                    self.game, search, model, config=algorithm_config))

            elif config.name == self.players.MANUAL.name:
                self.player_configs.append(self.players[config.name].player(
                    self.game, name=input("Input a player name: ")))
            else:
                self.player_configs.append(self.players[config.name].player(self.game))


def tournament_final(experiment: ExperimentConfig) -> None:

    # Initialize parametric players.
    for p in experiment.player_configs:
        if p.parametric:  # Load in latest model.
            p.model.load_checkpoint(*p.config.args.load_folder_file)

    results = list()
    for rep in trange(experiment.experiment_args.num_repeat, desc="Tourney repetition", file=sys.stdout):
        for players in tqdm(combinations(experiment.player_configs, experiment.game.n_players),
                            desc=f"Tourney {rep}", file=sys.stdout):
            if experiment.game.n_players == 1:
                pass
            else:
                arena = Arena(experiment.game, *players)

                all([x.bind_history(history=h) for x, h in zip(players, arena.trajectories)])

                results.append(arena.playGames(experiment.experiment_args.num_trials))

    print(results)


def tournament_pool(experiment: ExperimentConfig) -> None:
    pass
