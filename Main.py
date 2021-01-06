"""
File to perform small test runs on the codebase for both AlphaZero and MuZero.
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


def learnA0(g, a0_content, a0_run_name):
    net_args, args = a0_content.net_args, a0_content.args

    print("Testing:", ", ".join(a0_run_name.split("_")))

    net = ANet(g, net_args, a0_content.architecture)

    if args.load_model:
        net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = AlphaZeroCoach(g, net, args, a0_run_name)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    a0_content.to_json(f'{args.checkpoint}/{a0_run_name}.json')

    c.learn()


def learnM0(g, m0_content, m0_run_name):
    net_args, args = m0_content.net_args, m0_content.args

    print("Testing:", ", ".join(m0_run_name.split("_")))

    if args.latent_decoder:
        net = DMNet(g, net_args, m0_content.architecture)
    else:
        net = MNet(g, net_args, m0_content.architecture)

    if args.load_model:
        net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = MuZeroCoach(g, net, args, m0_run_name)

    m0_content.to_json(f'{args.checkpoint}/{m0_run_name}.json')

    c.learn()


def get_run_name(config_name, architecture, game):
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{config_name}_{architecture}_{game}_{time}"


def game_from_name(name):
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

    debugger.DEBUG_MODE = args.debug
    debugger.RENDER = args.render
    debugger.LOG_RATE = args.lograte

    if args.mode == "train":

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

            if content.algorithm == "ALPHAZERO":
                learnA0(game, content, run_name)
            elif content.algorithm == "MUZERO":
                learnM0(game, content, run_name)
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
