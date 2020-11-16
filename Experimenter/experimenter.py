"""
TODO: Development
"""
from dataclasses import dataclass
from itertools import combinations
import sys

from tqdm import trange

from Experimenter import Arena
import Agents
import Games

from utils import DotDict


@dataclass
class ExperimentConfig(object):

    def __init__(self, experiment_file: str):
        self.experiment_args = DotDict.from_json(experiment_file)
        self.game_config = None
        self.game = None
        self.player_configs = list()

    def construct(self) -> None:
        env = self.experiment_args.environment

        if env.name in Games.Games:
            self.game_config = Games.Games[env.name]
        else:
            raise NotImplementedError("Did not specify a valid environment.")

        self.game = self.game_config(**env.args)

        player_configs = self.experiment_args.players
        for config in player_configs:

            if config.name not in Agents.Players:
                raise NotImplementedError("Did not specify a valid environment.")

            self.player_configs.append(Agents.Players[config.name](self.game, config.config))


def tournament_final(experiment: ExperimentConfig) -> None:

    # Initialize parametric players.
    for p in experiment.player_configs:
        if p.parametric:  # Load in latest model.
            p.model.load_checkpoint(*p.args.args.load_folder_file)

    results = list()
    for _ in trange(experiment.experiment_args.num_repeat, desc="Tourney repetition", file=sys.stdout):
        for players in combinations(experiment.player_configs, experiment.game.n_players):
            print()
            if experiment.game.n_players == 1:
                arena = Arena(experiment.game, *players, *players)  # Duplicate players
                print("Playing:", arena.player1.name)
                trial_result = [arena.playGame(arena.player1) for _ in range(experiment.experiment_args.num_trials)]
                results.append(trial_result)
            else:
                arena = Arena(experiment.game, *players)
                print(f"Playing: {arena.player1.name} Vs {arena.player2.name}")
                trial_result = arena.playTurnGames(experiment.experiment_args.num_trials)
                results.append(trial_result)

    print(results)


def tournament_pool(experiment: ExperimentConfig) -> None:
    pass
