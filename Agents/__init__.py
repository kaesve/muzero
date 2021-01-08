"""
Initialization module to define constructor classes/ agent implementations in the 'Agents' scope.

To add a new neural network, add a key-argument to the AlphaZeroNetworks or MuZeroNetworks dictionary with as
value the class reference that constructs the neural network.
"""
from .GymNetwork import AlphaZeroGymNetwork, MuZeroGymNetwork
from .AtariNetwork import AlphaZeroAtariNetwork, MuZeroAtariNetwork
from .HexNetwork import AlphaZeroHexNetwork, MuZeroHexNetwork
from .Player import Player, ManualPlayer, RandomPlayer, DeterministicPlayer, \
    DefaultMuZeroPlayer, DefaultAlphaZeroPlayer, BlindMuZeroPlayer


# Add your AlphaZero neural network architecture here by referencing the imported Class with a string key.
AlphaZeroNetworks = {
    'Hex': AlphaZeroHexNetwork,
    'Othello': AlphaZeroHexNetwork,
    'Gym': AlphaZeroGymNetwork,
    "Atari": AlphaZeroAtariNetwork
}


# Add your MuZero neural network architecture here by referencing the imported Class with a string key.
MuZeroNetworks = {
    'Hex': MuZeroHexNetwork,
    'Othello': MuZeroHexNetwork,
    'Gym': MuZeroGymNetwork,
    'Atari': MuZeroAtariNetwork
}


# Add different agent implementations for interacting with environments.
Players = {
    "ALPHAZERO": DefaultAlphaZeroPlayer,
    "MUZERO": DefaultMuZeroPlayer,
    "BLIND_MUZERO": BlindMuZeroPlayer,
    "RANDOM": RandomPlayer,
    "DETERMINISTIC": DeterministicPlayer,
    "MANUAL": ManualPlayer
}
