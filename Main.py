"""
File to perform small test runs on the codebase for both AlphaZero and MuZero.
"""

# Bugfxing TF2?
# Prevent TF2 from hogging all the available VRAM when initializing?
# @url: https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Bugfxing TF2?

import json
from collections import deque

import numpy as np

from utils.storage import DotDict
from AlphaZero.Coach import Coach
from hex.HexGame import HexGame as Game
from hex.AlphaZeroModel.NNet import NNetWrapper as HexNet
from hex.MuZeroModel.NNet import NNetWrapper as MuHexNet
from MuZero.MuMCTS import MuZeroMCTS

ALPHAZERO_DEFAULTS = "Experimenter/Configs/SmallModel_AlphaZeroHex.json"
MUZERO_DEFAULTS = "Experimenter/MuZeroConfigs/default.json"

BOARD_SIZE = 5


def unpack_json(file: str):
    with open(file) as f:
        content = DotDict(json.load(f))
        name = content.name
        net_args = DotDict(content.net_args)
        args = DotDict(content.args)
    return name, net_args, args


def learnA0():
    name, net_args, args = unpack_json(ALPHAZERO_DEFAULTS)

    print("Testing:", name)

    g = Game(BOARD_SIZE)
    hex_net = HexNet(g, net_args)

    if args.load_model:
        hex_net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, hex_net, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    c.learn()


def learnM0():
    name, net_args, args = unpack_json(MUZERO_DEFAULTS)

    print("Testing:", name)

    g = Game(BOARD_SIZE)
    hex_net = MuHexNet(g, net_args)

    if args.load_model:
        hex_net.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    b = g.getInitBoard()
    g.display(b)
    # TODO: Important! Pad initial observations with a little noise to prevent a latent-state collapse to 0.
    history = deque([np.random.randn(*b.shape) * 1e-8] * (net_args.observation_length - 1) + [b], maxlen=net_args.observation_length)
    mcts = MuZeroMCTS(g, hex_net, args)

    player = 1
    for i in range(10):
        canon = g.getCanonicalForm(b, player)
        if player == 1:
            obs = np.array(history)

            pi_visit = mcts.getActionProb(obs, temp=1)
            a = np.random.choice(len(pi_visit), p=pi_visit)

            if g.getLegalMoves(canon, player)[a] == 0:
                print('illegal move')
                raise Exception("illegal move")
                a = np.argmax(g.getLegalMoves(canon, 1))
        else:
            a = np.argmax(g.getLegalMoves(canon, 1))

        b, player = g.getNextState(b, player, a)
        history.append(b)
        g.display(b)


if __name__ == "__main__":
    # learnA0()
    for _ in range(100):
        learnM0()
