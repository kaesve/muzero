"""
Main function of the codebase. This file is intended to call different parts of our pipeline based on console arguments.

To add new games to the pipeline, add your string_query-class constructor to the 'game_from_name' function.
"""
from datetime import datetime
import argparse
from tensorflow.python.client import device_lib

import utils.debugging as debugger
from utils.debugging import *
from utils.storage import DotDict
from utils.game_utils import DiscretizeAction

from AlphaZero.AlphaCoach import AlphaZeroCoach
from MuZero.MuCoach import MuZeroCoach

from AlphaZero.implementations.DefaultAlphaZero import DefaultAlphaZero as ANet
from MuZero.implementations.DefaultMuZero import DefaultMuZero as MNet
from MuZero.implementations.AEMuZero import DecoderMuZero as DMNet

from Games.hex.HexGame import HexGame
from Games.tictactoe.TicTacToeGame import TicTacToeGame
from Games.othello.OthelloGame import OthelloGame
from Games.gym.GymGame import GymGame
from Games.atari.AtariGame import AtariGame

import Experimenter
import Agents


def learnA0(g, a0_content: DotDict, a0_run_name: str) -> None:
    """
    Train an AlphaZero agent on the given environment with the specified configuration. If specified within the
    configuration file, the function will load in a previous model along with previously generated data.
    :param g: Game Instance of a Game class that implements environment logic. Train agent on this environment.
    :param a0_content: DotDict Data container with hyperparameters for AlphaZero
    :param a0_run_name: str Run name to store data by and annotate results.
    """
    print("Testing:", ", ".join(a0_run_name.split("_")))

    # Extract neural network and algorithm arguments separately
    net_args, alg_args = a0_content.net_args, a0_content.args
    net = ANet(g, net_args, a0_content.architecture)

    if alg_args.load_model:
        net.load_checkpoint(alg_args.load_folder_file[0], alg_args.load_folder_file[1])

    c = AlphaZeroCoach(g, net, alg_args, a0_run_name)
    if alg_args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    a0_content.to_json(f'{alg_args.checkpoint}/{a0_run_name}.json')

    c.learn()


def learnM0(g, m0_content: DotDict, m0_run_name: str) -> None:
    """
    Train an MuZero agent on the given environment with the specified configuration. If specified within the
    configuration file, the function will load in a previous model along with previously generated data.
    If specified, then the MuZero agent will also jointly train a state-transition decoder h^-1.
    :param g: Game Instance of a Game class that implements environment logic. Train agent on this environment.
    :param m0_content:
    :param m0_run_name:
    :return:
    """
    print("Testing:", ", ".join(m0_run_name.split("_")))

    # Extract neural network and algorithm arguments separately
    net_args, alg_args = m0_content.net_args, m0_content.args

    if alg_args.latent_decoder:  # This option for MuZero jointly trains a state-transition function decoder h^-1.
        net = DMNet(g, net_args, m0_content.architecture)
    else:
        net = MNet(g, net_args, m0_content.architecture)

    if alg_args.load_model:
        print("Load trainExamples from file")
        net.load_checkpoint(alg_args.load_folder_file[0], alg_args.load_folder_file[1])

    m0_content.to_json(f'{alg_args.checkpoint}/{m0_run_name}.json')

    c = MuZeroCoach(g, net, alg_args, m0_run_name)
    c.learn()


def get_run_name(config_name: str, architecture: str, game_name: str) -> None:
    """ Macro function to wrap various ModelConfig properties into a run name. """
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{config_name}_{architecture}_{game_name}_{time}"


def game_from_name(name: str):
    """
    Constructor function to yield a Game class by a query string.
    :param name: str Represents the name/ key of the environment to train on.
    :return: Game Instance of Game that contains the environment logic.
    """
    match_name = name.lower()

    if match_name == "hex":
        return HexGame(BOARD_SIZE)

    elif match_name == "tictactoe":
        return TicTacToeGame(BOARD_SIZE)

    elif match_name == "othello":
        return OthelloGame(BOARD_SIZE)

    elif match_name == "gym" or match_name == "cartpole":
        return GymGame("CartPole-v1")

    elif match_name == "pendulum":
        def discretize_wrapper(env):
            return DiscretizeAction(env, 15)
        return GymGame("Pendulum-v0", [discretize_wrapper])

    elif match_name.startswith("gym_"):
        return GymGame(name[len("gym_"):])

    elif match_name.startswith("atari_"):
        game_name = match_name[len("atari_"):]
        game_name = game_name.capitalize() + "NoFrameskip-v4"
        return AtariGame(game_name)

    # Add new environments here after defining them in Games.__init__.py
    # elif match_name.startswith("myenv"):
    #     return GymGame("myEnv")


if __name__ == "__main__":
    # Handle console arguments
    parser = argparse.ArgumentParser(description="A MuZero and AlphaZero implementation in Tensorflow.")

    mode_parsers = parser.add_subparsers(title="Modes")

    experiment_parser = mode_parsers.add_parser("experiment")
    experiment_parser.set_defaults(mode="experiment")

    train_parser = mode_parsers.add_parser("train")
    train_parser.set_defaults(mode="train")

    player_choices = ["manual", "random", "deterministic", "muzero", "alphazero"]
    play_parser = mode_parsers.add_parser("play")
    play_parser.set_defaults(mode="play", debug=True, render=True, lograte=0, gpu=0)
    play_parser.add_argument("--p1", choices=player_choices, default="manual", help="Player one")
    play_parser.add_argument("--p1_config", choices=player_choices, default=None, help="Config file for player one")
    play_parser.add_argument("--p2", choices=player_choices, default="manual", help="Player two")
    play_parser.add_argument("--p2_config", choices=player_choices, default=None, help="Config file for player two")

    # Single game modes
    for p in [train_parser, play_parser]:
        p.add_argument("--game", default="gym")
        p.add_argument("--boardsize", "-s", type=int, default=6, help="Board size (if relevant)")

    # Common arguments
    for p in [experiment_parser, train_parser]:
        # Debug settings
        p.add_argument("--debug", action="store_true", default=False, help="Turn on debug mode")
        p.add_argument("--lograte", type=int, default=1, help="Backprop logging frequency")
        p.add_argument("--render", action="store_true", default=False,
                       help="Render the environment during training and pitting")

        # Run configuration
        p.add_argument("--config", "-c", nargs="*", help="Path to config file", required=True)
        p.add_argument("--gpu", default=0, help="Set which device to use (-1 for CPU). Equivalent "
                                                "to/overrides the CUDA_VISIBLE_DEVICES environment variable.")
        p.add_argument("--run_name", default=False, help="Override the run name (will not be timestamped!)")

    args = parser.parse_args()
    # END Console arguments handling.

    # Set global debugging settings for monitoring purposes (can produce large tensorboard files!).
    debugger.DEBUG_MODE = args.debug
    debugger.RENDER = args.render
    debugger.LOG_RATE = args.lograte

    # Split up pipeline based on arguments
    if args.mode == "train":

        # Functionality to override parameters from within the console line. Use -c my_config.json override_config.json
        content = DotDict.from_json(args.config[0])
        for override in args.config[1:]:
            sub_config = DotDict.from_json(override)
            content.recursive_update(override)

        BOARD_SIZE = args.boardsize
        game = game_from_name(args.game)
        run_name = args.run_name if args.run_name else get_run_name(content.name, content.architecture, args.game)

        # Set up tensorflow backend.
        if int(args.gpu) >= 0:
            device = tf.DeviceSpec(device_type='GPU', device_index=int(args.gpu))
        else:
            device = tf.DeviceSpec(device_type='CPU', device_index=0)

        with tf.device(device.to_string()):
            switch = {'ALPHAZERO': learnA0, 'MUZERO': learnM0}
            if content.algorithm in switch:
                switch[content.algorithm](game, content, run_name)
            else:
                raise NotImplementedError(f"Cannot train on algorithm '{content.algorithm}'")

    elif args.mode == "experiment":
        b = Experimenter.ExperimentConfig(args.config[0])
        b.construct()

        print(f"Starting {b.type} experiment {b.name}, storing results in {b.output_directory}.")
        Experimenter.experiments[b.type](b)

    elif args.mode == "play":
        BOARD_SIZE = args.boardsize
        game = game_from_name(args.game)

        if args.p1.upper() not in Agents.Players:
            raise NotImplementedError(f"Did not specify a valid player one: {args.p1}")
        p1 = Agents.Players[args.p1.upper()](game, args.p1_config)

        if game.n_players == 1:
            arena = Experimenter.Arena(game, p1, None)
            arena.playGame(p1, True)
        elif game.n_players == 2:
            if args.p2.upper() not in Agents.Players:
                raise NotImplementedError(f"Did not specify a valid player two: {args.p2}")
            p2 = Agents.Players[args.p2.upper()](game)

            arena = Experimenter.Arena(game, p1, p2)
            arena.playTurnGame(p1, p2, True)
