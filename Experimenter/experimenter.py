"""
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import sys
import os
import typing
from datetime import datetime

from tqdm import trange
import numpy as np

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


def perform_tournament(experiment: ExperimentConfig, by_checkpoint: bool = True) -> None:
    """
    Helper function to unpack the player configs provided in the ExperimentConfig into a pool (list) of player-data
    tuples that is given to the tourney function. If 'by_checkpoint' is set to True, we check the directory
    of the provided model path and create individual players for each of the available model checkpoints.
    Otherwise we just take the (latest) model specified in the config.

    The experiment config must contain a 'checkpoint_resolution' integer argument to indicate a step to omit some of
    the checkpoints to reduce computation time --- i.e., use every 'checkpoint_resolution's model of x models.

    We expect model checkpoint files to be unaltered from the source code, meaning the format follows:
     - prefix_checkpoint_(int).pth.tar

    :param experiment: ExperimentConfig Contains the players to be pitted against each other.
    :param by_checkpoint: bool Whether to include every model checkpoint in the player pool (or just the specified one)
    """
    args = experiment.experiment_args  # Helper variable to reduce verbosity.
    # Collect player configurations
    player_checkpoint_pool = get_player_pool(experiment.player_configs, by_checkpoint=by_checkpoint,
                                             resolution=args.checkpoint_resolution)
    results = tourney(player_checkpoint_pool, experiment.game, args.num_repeat, args.num_trials, args.num_opponents)

    # Save results to output file.
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    data = DotDict({
        'results': results,
        'args': experiment.experiment_args
    })
    data.to_json(experiment.output_directory + f'{experiment.name}_{dt}.json')


def tourney(player_pool: typing.List, env: Games.Game, num_repeat: int,
            num_trials: int, num_opponents: typing.Optional[int] = None) -> typing.List[typing.Dict]:
    """
    Function to execute an exhaustive/ randomized tournament for the given player pool.
    This function will run 'nCr(len(player_pool), env.n_players) x num_repeat x num_trials' games.

    If players are parametric within the player pool, we cleverly re-use classes to reduce data from model weights.
    For 2 player games, we clone every distinct agent class reference in the player pool to prevent weight clashing.

    Data of the results are returned in the format:
    if env.n_players == 1:
        format = {
            "player": "{agent_class_name}_{agent_config_name}",
            "player_data": "None for non-parametric agents, (model_data_path, model_data_file) for parametric agents",
            "trial_result": "numpy array containing cumulative scores over num_trials games."
        }
    else:
        format = {
            "player1": "{agent1_class_name}_{agent1_config_name}",
            "player2": "{agent2_class_name}_{agent1_config_name}",
            "player1_data": "None for non-parametric agents, (model_data_path, model_data_file) for parametric agents",
            "player2_data": "None for non-parametric agents, (model_data_path, model_data_file) for parametric agents",
            "trial_result": "Dictionary with three integers indicating the wins, losses, and draws."
        }

    To reduce computation time, set num_opponents to a smaller value to randomly sample opponents, instead of
    exhaustively comparing all possible player combinations. (recommended for adversarial games when checking over a
    large number of checkpoint files).

    :param player_pool: List of tuples where each tuple is of the form (player, details)
    :param env: Games.Game implementation containing the logic of an environment.
    :param num_repeat: int Number of times to repeat the tourney
    :param num_trials: int Number of evaluation repetitions for each tourney.
    :param num_opponents: int If smaller than len(player_pool) we randomly (without replacement) select opponents.
    :return: List of dictionaries containing the results in the described format
    """

    def prepare(contestant: typing.Tuple[Agents.Player, str, str]) -> Agents.Player:
        """
        Unpack the player tuple into the Agents.Player interface, and load in requisite data.
        :param contestant: tuple of (Player base class, data path, data filename)
        :return: Agents.Player player interface to select actions given a state observation.
        """
        player_object, *content = contestant  # content is None for non-parametric agents.
        if player_object.parametric:
            player_object.model.load_checkpoint(*content)
        return player_object

    # We require a work around to copy player objects as we need to keep memory usage at a minimum
    # when testing multiple neural networks. We do this by reusing models and loading in weights.
    # The copy object is intended to prevent clashing of weights when agent class references are equal.
    copy_objects = {}
    if env.n_players > 1:
        for player, *_ in player_pool:
            if player not in copy_objects:
                copy_objects[player] = player.clone()

    schedule = list(combinations(player_pool, env.n_players))

    results = list()
    for _ in trange(num_repeat, desc="Tourney repetition", file=sys.stdout):
        if num_opponents < len(player_pool) and env.n_players > 1:
            # Create a new randomized schedule based on non-replacement random sampling.
            schedule = list()
            for p_i in range(len(player_pool)):
                opponents = np.random.choice([np.asarray(player_pool) != p_i], size=num_opponents, replace=False)
                schedule += [(player_pool[p_i], opponent) for opponent in opponents]

        print(f"Performing a total of {len(schedule)} tournaments in this tourney...")
        for i, player_data in enumerate(schedule):
            print(f"Tournament {i + 1} / {len(schedule)}. Playing:", player_data)
            if env.n_players == 1:
                player = prepare(*player_data)

                arena = Arena(env, player, player)  # Duplicate players
                trial_result = arena.playGames(num_trials, player)

                results.append({
                    'player': f'{type(player).__name__}_{player.name}',  # TODO: Bug tracking, .name is always empty.
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

                arena = Arena(env, player1, player2)
                win, loss, draw = arena.playTurnGames(num_trials)

                results.append({
                    'player1': f'{type(player1).__name__}_{player1.name}',
                    'player2': f'{type(player2).__name__}_{player2.name}',
                    'player1_data': player1_data[1:],
                    'player2_data': player2_data[1:],
                    'trial_result': {
                        'wins': win,
                        'losses': loss,
                        'draws': draw
                    }
                })

    return results
