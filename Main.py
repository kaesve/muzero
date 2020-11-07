"""
File to perform small test runs on the codebase for both AlphaZero and MuZero.
"""
from datetime import datetime

import utils.debugging as debugger
from utils.debugging import *
from utils.storage import DotDict

from AlphaZero.AlphaCoach import AlphaZeroCoach
from MuZero.MuCoach import MuZeroCoach

from AlphaZero.models.DefaultAlphaZero import DefaultAlphaZero as ANet
from MuZero.models.DefaultMuZero import DefaultMuZero as MNet

from Games.hex.HexGame import HexGame
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


def learnA0(g, config):
    content = DotDict.from_json(config)
    name, net_args, args = content.name, content.net_args, content.args

    print("Testing:", name)

    net = ANet(g, net_args, content.architecture)

    if args.load_model:
        net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = AlphaZeroCoach(g, net, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    content.to_json(f'out/AlphaZeroOut/{name}_{content.architecture}_{time}.json')

    c.learn()


def learnM0(g, config):
    content = DotDict.from_json(config)
    name, net_args, args = content.name, content.net_args, content.args

    print("Testing:", name)
    print("\n\n", content.architecture)

    net = MNet(g, net_args, content.architecture)

    if args.load_model:
        net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = MuZeroCoach(g, net, args)

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    content.to_json(f'out/MuZeroOut/{name}_{content.architecture}_{time}.json')

    c.learn()


if __name__ == "__main__":
    # Set debugging/ logging settings.
    debugger.DEBUG_MODE = True
    debugger.RENDER = False
    debugger.LOG_RATE = 1

    # learnA0(GymGame("CartPole-v1"), ALPHAZERO_DEFAULTS)
    # learnA0(HexGame(BOARD_SIZE), ALPHAZERO_BOARD)
    #
    # learnM0(HexGame(BOARD_SIZE), MUZERO_BOARD)
    learnM0(GymGame("CartPole-v1"), MUZERO_CARTPOLE)
    # learnM0(AtariGame('BreakoutNoFrameskip-v4'), MUZERO_ATARI)

    b = ExperimentConfig(MUZERO_RANDOM)
    # b = ExperimentConfig(ALPHAZERO_RANDOM)
    b.construct()
    print(b.game_config)
    print(b.player_configs)

    tournament_final(experiment=b)
