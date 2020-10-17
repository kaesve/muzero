"""
File to perform small test runs on the codebase for both AlphaZero and MuZero.
"""
# Suppress verbose warnings in stdout
import logging
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Bugfxing TF2?
# Prevent TF2 from hogging all the available VRAM when initializing?
# @url: https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Bugfxing TF2?

from utils.storage import DotDict
from AlphaZero.Coach import Coach
from Games.hex.HexGame import HexGame
from Games.hex.AlphaZeroModel.NNet import NNetWrapper as HexNet
from Games.hex.MuZeroModel.NNet import NNetWrapper as MuHexNet
from Games.atari.AtariGame import AtariGame
from Games.atari.MuZeroModel.NNet import NNetWrapper as MuAtariNet
from MuZero.MuCoach import MuZeroCoach
from Experimenter.experimenter import ExperimentConfig, tournament_final

ALPHAZERO_DEFAULTS = "Experimenter/AlphaZeroConfigs/default.json"
MUZERO_DEFAULTS = "Experimenter/MuZeroConfigs/default.json"

MUZERO_RANDOM = "Experimenter/JobConfigs/Tourney_Hex_MuZeroVsRandom.json"

BOARD_SIZE = 5


def learnA0():
    content = DotDict.from_json(ALPHAZERO_DEFAULTS)
    name, net_args, args = content.name, content.net_args, content.args

    print("Testing:", name)

    g = HexGame(BOARD_SIZE)
    hex_net = HexNet(g, net_args)

    if args.load_model:
        hex_net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, hex_net, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    c.learn()


def learnM0(g, Net):
    content = DotDict.from_json(MUZERO_DEFAULTS)
    name, net_args, args = content.name, content.net_args, content.args

    print("Testing:", name)

    net = Net(g, net_args)

    if args.load_model:
        net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = MuZeroCoach(g, net, args)

    c.learn()


if __name__ == "__main__":
    # learnA0()
    learnM0(HexGame(BOARD_SIZE), MuHexNet)
    # learnM0(AtariGame("BreakoutNoFrameskip-v4"), MuAtariNet)
    
    # b = ExperimentConfig(MUZERO_RANDOM)
    # b.construct()
    # print(b.game_config)
    # print(b.player_configs)

    # tournament_final(experiment=b)
