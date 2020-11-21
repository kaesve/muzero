from utils.storage import DotDict
from Games.hex.HexGame import HexGame as Game

from Games.hex.legacy.hex_policies import *
from Games.hex.legacy.experimenter import *

import time
import datetime

import numpy as np
import trueskill


def progression_tournament():
    """

    """

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
            'net_id': 2,  # Final net2net agent is a deep AlphaZeroModel.
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
        'shallow': {  # second shallow AlphaZeroModel
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
            config_net_args = DotDict(net_args.copy())

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
