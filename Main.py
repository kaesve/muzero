"""
File to perform small test runs on the codebase for both AlphaZero and MuZero.
"""
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

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

ALPHAZERO_DEFAULTS = "Configurations/ModelConfigs/AlphazeroCartpole.json"
ALPHAZERO_BOARD = "Configurations/ModelConfigs/AlphazeroHex.json"
MUZERO_DEFAULTS = "Configurations/ModelConfigs/MuzeroCartpole.json"
MUZERO_BOARD = "Configurations/ModelConfigs/MuzeroHex.json"

MUZERO_RANDOM = "Configurations/JobConfigs/Tourney_Hex_MuZeroVsRandom.json"

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

    c.learn()


def learnM0(g, config):
    content = DotDict.from_json(config)
    name, net_args, args = content.name, content.net_args, content.args

    print("Testing:", name)

    net = MNet(g, net_args, content.architecture)

    if args.load_model:
        net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = MuZeroCoach(g, net, args)

    c.learn()


if __name__ == "__main__":
    # Set debugging/ logging settings.
    debugger.DEBUG_MODE = True
    debugger.LOG_RATE = 1

    learnA0(GymGame("CartPole-v1"), ALPHAZERO_DEFAULTS)

    # learnM0(HexGame(BOARD_SIZE), MUZERO_BOARD)
    learnM0(GymGame('CartPole-v1'), MUZERO_DEFAULTS)

    # b = ExperimentConfig(MUZERO_RANDOM)
    # b.construct()
    # print(b.game_config)
    # print(b.player_configs)

    # tournament_final(experiment=b)
