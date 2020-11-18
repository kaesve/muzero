"""
Initialization module to define constructor classes/ agent implementations in the 'Agents' scope.
"""
from .GymNetwork import AlphaZeroGymNetwork, MuZeroGymNetwork
from .AtariNetwork import AlphaZeroAtariNetwork, MuZeroAtariNetwork
from .HexNetwork import AlphaZeroHexNetwork, MuZeroHexNetwork
from .Player import ManualPlayer, RandomPlayer, DeterministicPlayer, MuZeroPlayer, DefaultAlphaZeroPlayer


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
    "MUZERO": MuZeroPlayer,
    "RANDOM": RandomPlayer,
    "DETERMINISTIC": DeterministicPlayer,
    "MANUAL": ManualPlayer
}
