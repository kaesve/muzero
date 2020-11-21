"""
"""
from dataclasses import dataclass
from itertools import combinations
import sys
import os
import typing
from datetime import datetime

from tqdm import trange

from Experimenter import Arena
import Agents
import Games

from utils import DotDict
from utils.experimenter_utils import create_parameter_grid, get_player_pool



@dataclass
class ExperimentConfig(object):
    """
    Class to store and unpack data for performing experiments.
    This data structure should only provide an interface for functions to use trained models
    or player and environment interfaces.
    """

    def __init__(self, experiment_file: str) -> None:
        """
        Initialize the experiment data container using a string path to a .json settings file.
        Not all variables are initialized directly (require an explicit call to construct) seeing as
        this may bloat the memory with large models when performing a large number of experiments sequentially.

        :param experiment_file: str Path to .json file containing experiment details.
        """
        self.experiment_args = DotDict.from_json(experiment_file)
        self.type = self.experiment_args.experiment
        self.name = self.experiment_args.name

        self.output_directory = self.output_directory = f'./out/{self.experiment_args.output_dir}/'
        self.game_config = None
        self.game = None
        self.ablation_base = None
        self.ablation_grid = None
        self.player_configs = list()

    def construct(self) -> None:
        """
        Parse provided arguments to this data-container by the environment interface and preparing an output path.

        Dependent on the content of the experiment argument file, also initialize player interface and/ or
        unpack a given ablation configuration into a parameter-grid.
        """
        env = self.experiment_args.environment

        if env.name in Games.Games:
            self.game_config = Games.Games[env.name]
        else:
            raise NotImplementedError(f"Did not specify a valid environment: {env.name}")

        self.game = self.game_config(**env.args)

        if 'players' in self.experiment_args:
            player_configs = self.experiment_args.players
            for config in player_configs:

                if config.name not in Agents.Players:
                    raise NotImplementedError(f"Did not specify a valid player: {config.name}")

                self.player_configs.append(Agents.Players[config.name](self.game, config.config))

        if 'ablations' in self.experiment_args:
            self.ablation_base = self.experiment_args.ablations.base
            self.ablation_grid = create_parameter_grid(self.experiment_args.ablations.content)

        # Create output directory if note exists.
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)


def run_ablations(experiment: ExperimentConfig) -> None:
    pass


def tournament_final(experiment: ExperimentConfig) -> None:
    """
    Helper function to unpack the player configs provided in the ExperimentConfig into a pool (list) of player-data
    tuples that is given to the tourney function. The resulting data from the tourney is stored by this function.
    :param experiment: ExperimentConfig Contains the players to be pitted against each other.
    """
    player_pool = get_player_pool(experiment.player_configs)
    results = tourney(experiment, player_pool)

    # Save results along with program arguments.
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    data = DotDict({
        'results': results,
        'args': experiment.experiment_args
    })
    data.to_json(experiment.output_directory + f'{experiment.name}_{dt}.json')


def tournament_pool(experiment: ExperimentConfig) -> None:
    """
    Helper function to unpack the player configs provided in the ExperimentConfig into a pool (list) of player-data
    tuples that is given to the tourney function. The difference from tournament_final is that we check the directory
    of the provided model path and create individual players for each of the available model checkpoints.

    The experiment config must contain a 'checkpoint_resolution' integer argument to indicate a step to omit some of
    the checkpoints to reduce computation time --- i.e., use every 'checkpoint_resolution's model of x models.

    We expect model checkpoint files to be unaltered from the source code, meaning the format follows:
     - prefix_checkpoint_(int).pth.tar

    The resulting data from the tourney is stored by this function.
    :param experiment: ExperimentConfig Contains the players to be pitted against each other.
    """
    # Collect player configurations of the form
    player_checkpoint_pool = get_player_pool(experiment.player_configs, by_checkpoint=True,
                                             resolution=experiment.experiment_args.checkpoint_resolution)
    results = tourney(experiment, player_checkpoint_pool)

    # Save results along with program arguments.
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    data = DotDict({
        'results': results,
        'args': experiment.experiment_args
    })
    data.to_json(experiment.output_directory + f'{experiment.name}_{dt}.json')


def tourney(experiment: ExperimentConfig, player_pool: typing.List) -> typing.Tuple[typing.Dict]:

    def prepare(contestant: typing.Tuple[Agents.Player, str, str]) -> Agents.Player:

        player_object, *content = contestant
        if player_object.parametric:
            player_object.model.load_checkpoint(*content)

        return player_object

    # We require a work around to copy player objects as we need to keep memory usage at a minimum
    # when testing multiple neural networks. We do this by reusing models and loading in weights.
    copy_objects = {}
    for player, *_ in player_pool:
        if player not in copy_objects:
            copy_objects[player] = player.clone()

    schedule = list(combinations(player_pool, experiment.game.n_players))
    print(f"Performing a total of {len(schedule)} tournaments per repetition.")

    results = list()
    for _ in trange(experiment.experiment_args.num_repeat, desc="Tourney repetition", file=sys.stdout):
        for i, player_data in enumerate(schedule):
            print(f"Tournament {i+1} / {len(schedule)}. Playing:", player_data)
            if experiment.game.n_players == 1:
                player = prepare(*player_data)

                arena = Arena(experiment.game, player, player)  # Duplicate players
                trial_result = arena.playGames(experiment.experiment_args.num_trials, player)

                results.append({
                    'player': type(player).__name__,
                    'player_data': player_data[1:],
                    'trial_result': trial_result
                })
            else:
                player1_data, player2_data = player_data

                player1 = prepare(player1_data)
                if player1_data[0] is player2_data[0]:
                    player2 = prepare((copy_objects[player1_data[0]], player2_data[1], player2_data[2]))
                else:
                    player2 = prepare(player2_data)

                arena = Arena(experiment.game, player1, player2)
                win, loss, draw = arena.playTurnGames(experiment.experiment_args.num_trials)

                results.append({
                    'player1': type(player1).__name__,
                    'player2': type(player2).__name__,
                    'player1_data': player1_data[1:],
                    'player2_data': player2_data[1:],
                    'trial_result': {
                        'wins': win,
                        'losses': loss,
                        'draws': draw
                    }
                })

    return results
