"""
File to perform small test runs on the codebase for both AlphaZero and MuZero.
"""

from utils.storage import DotDict
from AlphaZero.Coach import Coach
from hex.HexGame import HexGame as Game
from hex.model.NNet import NNetWrapper as HexNet


args = DotDict({
    'numIters': 3,            # (1000)
    'numEps': 10,             # Number of complete self-play games to simulate during a new iteration.  (100)
    'tempThreshold': 15,      #
    'updateThreshold': 0.6,   # Pitting: new neural net will be accepted if win ratio exceeds this threshold.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 25,        # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,        # Number of games to play during arena play to determine if new net will be accepted. (40)
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

net_args = DotDict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 256,  # Default 512. DownScaled 256.
    'default': False,     # True: Use large alpha-zero-general model. False use other models specified by transfer
    'transfer': 0         # 0: shallow model 1: medium model 2: deep model  << alpha-zero-general model
})

BOARD_SIZE = 5


def learnA0():
    """
    Train a network with the given configuration on the game of Hex.

    Paths are set relatively to the current working directory.

    :param manual: Set to False to utilize the baseline
    :param model: int if manual = True: 0 = shallow, 1 = medium, 2 = deep
    :param debug: Set to True to set the disk-path to a debugging folder.
    """
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
    g = Game(BOARD_SIZE)


if __name__ == "__main__":
    learnA0()
