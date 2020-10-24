"""
File to perform small test runs on the codebase for both AlphaZero and MuZero.
"""
# Suppress verbose warnings in stdout
from utils.debugging import *

from utils.storage import DotDict
from AlphaZero.Coach import Coach
from Games.hex.HexGame import HexGame
from Games.hex.AlphaZeroModel.NNet import NNetWrapper as HexNet
from Games.hex.MuZeroModel.NNet import NNetWrapper as MuHexNet
from Games.gym.GymGame import GymGame
from Games.gym.MuZeroModel.NNet import NNetWrapper as MuGymNet
from Games.atari.AtariGame import AtariGame
from Games.atari.MuZeroModel.NNet import NNetWrapper as MuAtariNet
from MuZero.MuCoach import MuZeroCoach
from Experimenter.experimenter import ExperimentConfig, tournament_final

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

ALPHAZERO_DEFAULTS = "Experimenter/AlphaZeroConfigs/singleplayergames.json"
MUZERO_DEFAULTS = "Experimenter/MuZeroConfigs/singleplayergames.json"
MUZERO_BOARD = "Experimenter/MuZeroConfigs/boardgames.json"

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


def learnM0(g, Net, config):
    content = DotDict.from_json(config)
    name, net_args, args = content.name, content.net_args, content.args

    print("Testing:", name)

    net = Net(g, net_args)

    if args.load_model:
        net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = MuZeroCoach(g, net, args)

    c.learn()


if __name__ == "__main__":
    # learnA0()
    # learnM0(HexGame(BOARD_SIZE), MuHexNet, MUZERO_BOARD)
    learnM0(GymGame('CartPole-v1'), MuGymNet, MUZERO_DEFAULTS)
    # learnM0(AtariGame("BreakoutNoFrameskip-v4"), MuAtariNet)
    
    # b = ExperimentConfig(MUZERO_RANDOM)
    # b.construct()
    # print(b.game_config)
    # print(b.player_configs)

    # tournament_final(experiment=b)
