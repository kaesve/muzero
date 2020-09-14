"""
Main python file to control the flow of training and pitting AlphaZero agents
on the game of Hex.

To perform all steps described in the report, paths to models have to be edited manually.
It was not feasible to automate this due to frequent system crashes (LIACS Lab).

Due to the crashes, it may also be neccesary to manually edit the code for the `learn_net2net'
function and the `learn' function calls in order to load in a previous checkpoint.

:version: FINAL
:date: 15-05-2020
:author: Joery de Vries
"""

# TENSORFLOW GPU CONFIGURATION
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Edit to select GPU!
# TENSORFLOW GPU CONFIGURATION


from Coach import Coach
from hex.HexGame import HexGame as Game
from hex.model.NNet import NNetWrapper as nn
from hex.model.net2net import shallow_to_medium, medium_to_deep
from utils import *

from hex.src_joery.hex_policies import *
from hex.src_joery.hex_player import *
from hex.src_joery.hex_utils import *
from hex.src_joery.hex_heuristics import *
from hex.src_joery.experimenter import *

import time
import datetime

import numpy as np
import trueskill


# Bugfxing TF2?
# Prevent TF2 from hogging all the available VRAM when initializing?
# @url: https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Bugfxing TF2?


args = dotdict({
    'numIters': 300,  # (1000)
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.  (100)
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted. (40)
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

net_args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 256,  # Default 512. DownScaled 256.
    'default': True,     # True: Use large alpha-zero-general model. False use other models specified by transfer
    'transfer': 0         # 0: shallow model 1: medium model 2: deep model  << alpha-zero-general model
})


BOARD_SIZE = 7


def learn(manual=True, model=0, debug=False):
    """
    Train a network with the given configuration on the game of Hex.

    Paths are set relatively to the current working directory.

    :param manual: Set to False to utilize the baseline
    :param model: int if manual = True: 0 = shallow, 1 = medium, 2 = deep
    :param debug: Set to True to set the disk-path to a debugging folder.
    """
    g = Game(BOARD_SIZE)

    if manual:
        net_args.default = False
        net_args.transfer = model

        if net_args.transfer == 0:
            args.checkpoint = './temp/hex/small/'
            args.load_folder_file = ('./temp/hex/small/', 'best.pth.tar')
        elif net_args.transfer == 1:
            args.checkpoint = './temp/hex/medium/'
            args.load_folder_file = ('./temp/hex/medium/', 'best.pth.tar')
        elif net_args.transfer == 2:
            args.checkpoint = './temp/hex/large/'
            args.load_folder_file = ('./temp/hex/large/', 'best.pth.tar')
    else:
        net_args.num_channels = 512  # A0G default.

    if debug:
        args.checkpoint = './temp/hex/debug/'
        args.load_folder_file = ('./temp/hex/debug/', 'best.pth.tar')

    nnet = nn(g, net_args)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    c.learn()


def learn_net2net():
    """
    Train the Net2Net agent on the game of Hex.

    The agent is trained for 100 iterations using the small model.
    100 iterations using the medium model.
    and 100 iterations using the deep model.

    The backend is based on:
    https://keras.io/examples/mnist_net2net/
    """
    g = Game(BOARD_SIZE)
    verbose_net2net = True

    # START shallow
    # Override default arguments.
    shallow_args = dotdict(net_args.copy())
    shallow_args.default = False
    shallow_args.transfer = 0
    shallow_args.num_channels = 256
    shallow_nnet = nn(g, shallow_args)

    # Coach for shallow model
    args.numIters = 100
    args.checkpoint = './temp/hex/transfer/shallow/'
    args.load_folder_file = ('./temp/hex/transfer/shallow/', 'best.pth.tar')
    shallow_nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, shallow_nnet, args)

    # Learn Shallow
    c.learn()

    # START medium
    medium_args = dotdict(shallow_args.copy())
    medium_args.transfer += 1
    medium_nnet = nn(g, medium_args)

    # net2net
    boards = [[entry[0] for entry in episode_data] for episode_data in c.trainExamplesHistory]
    boards = np.vstack([np.array(board).reshape((-1, BOARD_SIZE, BOARD_SIZE)) for board in boards])
    shallow_to_medium(shallow_nnet.playerNet.model, medium_nnet.playerNet.model, boards, verbose_net2net)

    # Coach for the medium model
    args.numIters = 100
    args.checkpoint = './temp/hex/transfer/medium/'
    args.load_folder_file = ('./temp/hex/transfer/medium/', 'best.pth.tar')
    medium_nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, medium_nnet, args)

    # Learn medium
    c.learn()

    # START DEEP
    deep_args = dotdict(medium_args.copy())
    deep_args.transfer += 1
    deep_nnet = nn(g, deep_args)

    # net2net
    boards = [[entry[0] for entry in episode_data] for episode_data in c.trainExamplesHistory]
    boards = np.vstack([np.array(board).reshape((-1, BOARD_SIZE, BOARD_SIZE)) for board in boards])
    medium_to_deep(medium_nnet.playerNet.model, deep_nnet.playerNet.model, boards, verbose_net2net)

    # Coach for the deep model
    args.numIters = 100
    args.checkpoint = './temp/hex/transfer/deep/'
    args.load_folder_file = ('./temp/hex/transfer/deep/', 'best.pth.tar')
    deep_nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, deep_nnet, args)

    # Learn deep
    c.learn()
    # Done


def progression_tournament():  # TODO run.
    """

    """
    from tensorflow.keras import backend as K

    # Every configuration will play against 'resolution' uniformly random opponents.
    # This prevents combinatorial growth in the amount of games that need to be played.
    # However, this also decreases accuracy. Given N players, play N * k * r games where each
    # player will play at least k games.
    # It is recommended to use as default at least k=12! But this may be factors higher as
    # we're uniformly random constructing games, and not matching games by performance.
    # :see: https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
    k = 18  # Amount of opponents to play against per player.
    r = 4   # Amount of games to play per tourney

    # Dictionary to define the player pool for readability and storage.
    players = {
        'net2net_deep': {
            'path': '/data/jdevries/models/transfer/deep/',
            'net_default': False,
            'net_id': 2,  # Final net2net agent is a deep model.
            'checkpoints': {},
            'checkpoint_ratings': {}
        },
        'net2net_medium': {
            'path': '/data/jdevries/models/transfer/medium/',
            'net_default': False,
            'net_id': 1,
            'checkpoints': {},
            'checkpoint_ratings': {}
        },
        'net2net_small': {
            'path': '/data/jdevries/models/transfer/small/',
            'net_default': False,
            'net_id': 0,
            'checkpoints': {},
            'checkpoint_ratings': {}
        },
        'shallow': {  # second shallow model
            'path': '/data/jdevries/models/small/',
            'net_default': False,
            'net_id': 0,
            'checkpoints': {},
            'checkpoint_ratings': {}
        },
        'medium': {
            'path': '/data/jdevries/models/medium/',
            'net_default': False,
            'net_id': 1,
            'checkpoints': {},
            'checkpoint_ratings': {}
        },
        'deep': {
            'path': '/data/jdevries/models/large/',
            'net_default': False,
            'net_id': 2,
            'checkpoints': {},
            'checkpoint_ratings': {}
        },
        'alpha0general': {
            'path': '/data/jdevries/models/alphazero_general/',
            'net_default': True,
            'net_id': 0,
            'checkpoints': {},
            'checkpoint_ratings': {}
        }
    }

    g = Game(BOARD_SIZE)

    # List to functionally define the player pool. (constructing tourneys)
    player_pool = list()
    network_pool = dict()

    for player, config in players.items():
        files = os.listdir(config['path'])
        ids = [int(f.split('_')[0]) + int(f.strip('.pth.tar').split('_')[-1]) for f in files if 'best' not in f]

        config['checkpoints'] = dict(zip(ids, files))
        config['checkpoint_ratings'] = {id: trueskill.Rating() for id in ids}

        for f, id in zip(files, ids):
            config_args = args.copy()
            config_net_args = dotdict(net_args.copy())

            config_args['load_folder_file'] = (config['path'], f)

            config_net_args['default'] = config['net_default']
            config_net_args['transfer'] = config['net_id']
            config_net_args['num_channels'] = 512 if config['net_default'] else 256

            net_key = (config['net_default'], config['net_id'])
            if net_key not in network_pool:
                network_pool[net_key] = nn(g, config_net_args)

            player_pool.append({
                'name': player,
                'path': config['path'],
                'file': f,
                'id': id,
                'rating': config['checkpoint_ratings'][id],
                'args': config_args,
                'net_args': net_key
            })

    t0 = time.time()
    n_games = len(player_pool) * k

    # Play tournaments as to estimate TrueSkill
    for i, player in enumerate(player_pool):
        print("\nPitting Policy {} / {}\t".format(i+1, len(player_pool)), end='\n')
        player_nnet = network_pool[player['net_args']]               # Fetch architecture
        player_nnet.load_checkpoint(player['path'], player['file'])  # Fetch weights

        adversaries = np.delete(np.arange(len(player_pool)), i)
        opponent_ids = np.random.choice(adversaries, size=k, replace=False)
        opponents = [player_pool[i] for i in opponent_ids]

        policy = AlphaZeroPolicy(player['args']['cpuct'], player['args']['numMCTSSims'], player_nnet, BOARD_SIZE)

        for j, (id, opponent) in enumerate(zip(opponent_ids, opponents)):
            opponent_nnet = network_pool[opponent['net_args']]                 # Fetch architecture
            opponent_nnet.load_checkpoint(opponent['path'], opponent['file'])  # Fetch weights

            opponent_policy = AlphaZeroPolicy(opponent['args']['cpuct'], opponent['args']['numMCTSSims'], opponent_nnet, BOARD_SIZE)

            new_ratings = doubles_ratings(BOARD_SIZE, policy_list=[policy, opponent_policy], resolution=r,
                                      verbose=False, monitor=False, ratings=[player['rating'], opponent['rating']])

            player['rating'], opponent['rating'] = new_ratings

            eta_sec = (time.time() - t0) / (i * k + j+1) * (n_games - (i * k + j + 1))
            eta_dt = datetime.timedelta(seconds=eta_sec)
            print("Progression: {} / {}\tOpponent {} / {}\t, ETA: {}".format(
                i+1, len(player_pool), j+1, k, str(eta_dt)), end='\r')

    np.save('progression_tournament_trueskill', player_pool, allow_pickle=True)


def simple_tournament():
    """
    Performs the tournament between two equivalently trained shallow models
    the AlphaZero General model, an MCTS agent, and an IDTT agent.

    The agents were trained for 100 iterations in this tournament.

    Model paths are defined *Manually*.
    The parameters (net_args, args) should be kept at the source code's defaults.
    """
    g = Game(BOARD_SIZE)
    resolution = 12

    # Define configuration to correctly load in the players to be tested!
    shallow_args_1_net_args = dotdict(net_args.copy())
    shallow_args_2_net_args = dotdict(net_args.copy())
    alphazero_general_net_args = dotdict(net_args.copy())

    shallow_args_1_net_args.default = False
    shallow_args_1_net_args.num_channels = 256
    shallow_args_1_net_args.transfer = 0
    shallow_1_path = "./models/"
    shallow_1_filename = "shallow_model_1"

    shallow_args_2_net_args.default = False
    shallow_args_2_net_args.num_channels = 256
    shallow_args_2_net_args.transfer = 0
    shallow_2_path = "./models/"
    shallow_2_filename = "shallow_model_2"

    alphazero_general_net_args.default = True
    alphazero_general_net_args.num_channels = 512
    alphazero_general_path = "./models/"
    alphazero_general_filename = "alphazero_general"

    shallow_model_1 = nn(g, shallow_args_1_net_args)
    shallow_model_2 = nn(g, shallow_args_2_net_args)
    alphazero_general = nn(g, alphazero_general_net_args)

    shallow_model_1.load_checkpoint(shallow_1_path, shallow_1_filename)
    shallow_model_2.load_checkpoint(shallow_2_path, shallow_2_filename)
    alphazero_general.load_checkpoint(alphazero_general_path, alphazero_general_filename)
    # END Define configurations

    shallow1_alphazero = AlphaZeroPolicy(1, 25, shallow_model_1, BOARD_SIZE)
    shallow2_alphazero = AlphaZeroPolicy(1, 25, shallow_model_2, BOARD_SIZE)
    general_alphazero = AlphaZeroPolicy(1, 25, alphazero_general, BOARD_SIZE)

    shallow1_alphazero_high = AlphaZeroPolicy(1, 250, shallow_model_1, BOARD_SIZE)
    shallow2_alphazero_high = AlphaZeroPolicy(1, 250, shallow_model_2, BOARD_SIZE)
    general_alphazero_high = AlphaZeroPolicy(1, 250, alphazero_general, BOARD_SIZE)

    mcts_low = MCTSPolicy(1, 100, memorize=False, monitor=True)
    mcts_medium = MCTSPolicy(1, 1_000, memorize=False, monitor=True)
    mcts_high = MCTSPolicy(1, 10_000, memorize=False, monitor=True)

    idtt_low = MinimaxPolicy(DijkstraHeuristic(), itd=True, transpositions=True, budget=500)
    idtt_high = MinimaxPolicy(DijkstraHeuristic(), itd=True, transpositions=True, budget=3_000)

    policies = [shallow1_alphazero, shallow2_alphazero, general_alphazero,
                shallow1_alphazero_high, shallow2_alphazero_high, general_alphazero_high,
                mcts_low, mcts_medium, mcts_high, idtt_low, idtt_high]
    names = ['AlphaZero Shallow 1 $N=25$', 'AlphaZero Shallow 2 $N=25$', 'AlphaZero General $N=25$',
             'AlphaZero Shallow 1 $N=250$', 'AlphaZero Shallow 2 $N=250$', 'AlphaZero General $N=250$',
             'MCTS $N=10^2$', 'MCTS $N=10^3$', 'MCTS $N=10^4$', 'IDTT $N=5\cdot 10^2$', 'IDTT $N=3\cdot 10^3$']

    ratings, history = doubles_ratings(BOARD_SIZE, policy_list=policies,
                    resolution=resolution, verbose=True, monitor=True, name_list=names)

    np.save('results/ratings_history_tournament', [ratings, history], allow_pickle=True)
    plot_history(history, names, 'results/ratings_history_tournament.pdf')


def final_tournament():
    """
    Performs the tournament between the five fully trained AlphaZero agents.
    The agents were trained for 300 iterations in this tournament.
    MCTS and IDTT are used as baselines.

    Model paths are defined *Manually*.
    The parameters (net_args, args) should be kept at the source code's defaults.
    """
    g = Game(BOARD_SIZE)
    resolution = 12

    # Define configuration to correctly load in the players to be tested!
    shallow_args_net_args = dotdict(net_args.copy())
    medium_args_net_args = dotdict(net_args.copy())
    deep_args_net_args = dotdict(net_args.copy())
    net2net_net_args = dotdict(net_args.copy())
    alphazero_general_net_args = dotdict(net_args.copy())

    shallow_args_net_args.default = medium_args_net_args.default = \
        deep_args_net_args.default = net2net_net_args.default = False
    shallow_args_net_args.num_channels = 256
    medium_args_net_args.num_channels = 256
    deep_args_net_args.num_channels = 256
    net2net_net_args.num_channels = 256
    shallow_args_net_args.transfer = 0
    medium_args_net_args.transfer = 1
    deep_args_net_args.transfer = net2net_net_args.transfer = 2

    alphazero_general_net_args.default = True
    alphazero_general_net_args.num_channels = 512

    path = "./models/final/"
    shallow_filename = "small_model"
    medium_filename = "medium_model"
    large_filename = "large_model"
    net2net_filename = "net2net"
    alphazero_general_filename = "alphazero_general"

    shallow = nn(g, shallow_args_net_args)
    medium = nn(g, medium_args_net_args)
    deep = nn(g, deep_args_net_args)
    net2net = nn(g, net2net_net_args)
    alphazero_general = nn(g, alphazero_general_net_args)

    shallow.load_checkpoint(path, shallow_filename)
    medium.load_checkpoint(path, medium_filename)
    deep.load_checkpoint(path, large_filename)
    net2net.load_checkpoint(path, net2net_filename)
    alphazero_general.load_checkpoint(path, alphazero_general_filename)
    # END Define configurations

    shallow_alphazero = AlphaZeroPolicy(1, 25, shallow, BOARD_SIZE)
    medium_alphazero = AlphaZeroPolicy(1, 25, medium, BOARD_SIZE)
    deep_alphazero = AlphaZeroPolicy(1, 25, deep, BOARD_SIZE)
    net2net_alphazero = AlphaZeroPolicy(1, 25, net2net, BOARD_SIZE)
    general_alphazero = AlphaZeroPolicy(1, 25, alphazero_general, BOARD_SIZE)

    mcts_high = MCTSPolicy(1, 10_000, memorize=False, monitor=True)
    idtt_high = MinimaxPolicy(DijkstraHeuristic(), itd=True, transpositions=True, budget=3_000)

    policies = [shallow_alphazero, medium_alphazero, deep_alphazero, net2net_alphazero,
                general_alphazero, mcts_high, idtt_high]
    names = ['AlphaZero Shallow', 'AlphaZero Medium', 'AlphaZero Deep', 'AlphaZero Net2Net',
             'AlphaZero General', 'MCTS $N=10^4$', r'IDTT $N=3 \cdot 10^3$']

    ratings, history = doubles_ratings(BOARD_SIZE, policy_list=policies,
                                       resolution=resolution, verbose=True, monitor=True, name_list=names)

    np.save('results/final_ratings_history_tournament', [ratings, history], allow_pickle=True)
    plot_history(history, names, 'results/final_ratings_history_tournament.pdf')


def plot_history(history, labels, filename):
    """
    Create a simple TrueSkill convergence plot annotated with labels.
    :param history: History object of a tournament procedure.
    :param labels: list of strings that provide labels to the history's objects
    :param filename: str Path to the outputfile.
    """
    import matplotlib.pyplot as plt

    for player, skill in history.items():
        means = np.array([s[0] for s in skill])
        sigmas = np.array([s[1] for s in skill])

        plt.plot(range(len(skill)), means, label=labels[player])
        plt.fill_between(range(len(skill)), means + sigmas, means - sigmas, alpha=0.1)

    plt.ylabel(r"TrueSkill ($\mu \pm \sigma$)")
    plt.xlabel(r"Amount of games ($n$)")
    plt.legend()
    plt.savefig(filename, format='pdf')
    plt.close()


if __name__ == "__main__":
    learn(True, model=0, debug=False)  # Train the Shallow Model
    # learn(True, model=1, debug=False)  # Train the Medium Model
    # learn(True, model=2, debug=False)  # Train the Deep Model
    # learn(False)                       # Train the Baseline Alpha0General model
    # learn_net2net()                    # Train the Net2Net model (shallow -> medium -> deep)
    # simple_tournament()                # Perform the interim tourney
    # final_tournament()                 # Perform the final tourney
    # progression_tournament()           # Perform the learning curve estimation tourney
    pass
