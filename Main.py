"""
File to perform small test runs on the codebase for both AlphaZero and MuZero.
"""
from datetime import datetime
import argparse

import utils.debugging as debugger
from utils.debugging import *
from utils.storage import DotDict

from AlphaZero.AlphaCoach import AlphaZeroCoach
from MuZero.MuCoach import MuZeroCoach

from AlphaZero.models.DefaultAlphaZero import DefaultAlphaZero as ANet
from MuZero.models.DefaultMuZero import DefaultMuZero as MNet

from Games.hex.HexGame import HexGame
from Games.othello.OthelloGame import OthelloGame
from Games.gym.GymGame import GymGame
from Games.atari.AtariGame import AtariGame

from Experimenter.experimenter import ExperimentConfig, tournament_final

ALPHAZERO_DEFAULTS = "Configurations/ModelConfigs/AlphazeroCartpole.json"
ALPHAZERO_BOARD = "Configurations/ModelConfigs/AlphazeroHex.json"
MUZERO_CARTPOLE = "Configurations/ModelConfigs/MuzeroCartpole.json"
MUZERO_ATARI = "Configurations/ModelConfigs/MuzeroAtari.json"
MUZERO_BOARD = "Configurations/ModelConfigs/MuzeroHex.json"

MUZERO_RANDOM = "Configurations/JobConfigs/Tourney_Hex_MuZeroVsRandom.json"
ALPHAZERO_RANDOM = "Configurations/JobConfigs/Tourney_Hex_AlphaZeroVsRandom.json"

BOARD_SIZE = 5


def get_run_name(config_name, architecture, game):
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{config_name}_{architecture}_{game}_{time}"

def learnA0(g, content, run_name):
    net_args, args = content.net_args, content.args

    print("Testing:", ", ".join(run_name.split("_")))

    net = ANet(g, net_args, content.architecture)

    if args.load_model:
        net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = AlphaZeroCoach(g, net, args, run_name)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    content.to_json(f'out/AlphaZeroOut/{run_name}.json')

    c.learn()


def learnM0(g, content, run_name):
    net_args, args = content.net_args, content.args

    print("Testing:", ", ".join(run_name.split("_")))

    net = MNet(g, net_args, content.architecture)

    if args.load_model:
        net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = MuZeroCoach(g, net, args, run_name)

    content.to_json(f'out/MuZeroOut/{run_name}.json')

    c.learn()


def game_from_name(name):
    match_name = name.lower()

    if match_name == "hex":
        return HexGame(BOARD_SIZE)
    elif match_name == "othello":
        return OthelloGame(BOARD_SIZE)
    elif match_name == "gym":
        return GymGame("CartPole-v1")
    elif match_name.startswith("gym_"):
        return GymGame(name[len("gymS"):])
    elif match_name.startswith("atari_"):
        game_name = match_name[len("atari_"):]
        game_name = game_name.capitalize() + "NoFrameskip-v4"
        return AtariGame(game_name)


if __name__ == "__main__":
    # learnA0(GymGame("CartPole-v1"), ALPHAZERO_DEFAULTS)
    # learnA0(HexGame(BOARD_SIZE), ALPHAZERO_BOARD)
    #
    debugger.DEBUG_MODE = True
    content = DotDict.from_json(MUZERO_CARTPOLE)
    # game = HexGame(BOARD_SIZE)
    # learnM0(game, content)
    learnM0(GymGame("CartPole-v1"), content)
    # learnM0(AtariGame('BreakoutNoFrameskip-v4'), MUZERO_ATARI)

    # b = ExperimentConfig(MUZERO_RANDOM)
    # b = ExperimentConfig(ALPHAZERO_RANDOM)
    # b = ExperimentConfig(args.config)
    # b.construct() 
    # print(b.game_config)
    # print(b.player_configs)

    # tournament_final(experiment=b)

    parser = argparse.ArgumentParser(description="A MuZero and AlphaZero implementation in Tensorflow.")

    parser.add_argument("--debug", action="store_true", default=False, help="Turn on debug mode")
    parser.add_argument("--lograte", type=int, default=1, help="Backprop logging frequency")
    parser.add_argument("--render", action="store_true", default=False, help="Render the environment during training and pitting")

    modes = [ "train", "experiment" ]
    parser.add_argument("--mode", "-m", choices=modes, default="experiment")
    parser.add_argument("--config", "-c", help="Path to config file", required=True)
    parser.add_argument("--boardsize", "-s", type=int, default=BOARD_SIZE, help="Board size (if relevant)")

    mode_parsers = parser.add_subparsers(title="Modes")
    
    experiment_parser = mode_parsers.add_parser("experiment")
    experiment_parser.set_defaults(mode="experiment")

    train_parser = mode_parsers.add_parser("train")
    train_parser.set_defaults(mode="train")
    train_parser.add_argument("--game", default="gym")

    args = parser.parse_args()

    print("DEBUG IS ", args.debug)

    debugger.DEBUG_MODE = args.debug
    debugger.RENDER = args.render
    debugger.LOG_RATE = args.lograte

    BOARD_SIZE = args.boardsize
    
    if args.mode == "train":
        content = DotDict.from_json(args.config)
        game = game_from_name(args.game)
        run_name = get_run_name(content.name, content.architecture, args.game)

        if content.algorithm == "ALPHAZERO":
            learnA0(game, content, run_name)
        elif content.algorithm == "MUZERO":
            learnM0(game, content, run_name)
        else:
            raise NotImplementedError(f"Cannot train on algorithm '{content.algorithm}'")

    elif args.mode == "experiment":
        b = ExperimentConfig(args.config)
        b.construct() 
        print(b.game_config)
        print(b.player_configs)

        tournament_final(experiment=b)





