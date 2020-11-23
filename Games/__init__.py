"""
Initialization module to define the available Game implementations in the 'Games' scope.
"""
from .Game import Game
from .atari.AtariGame import AtariGame
from .gym.GymGame import GymGame
from .gym.ImageGymGame import ImageGymGame
from .hex.HexGame import HexGame
from .tictactoe.TicTacToeGame import TicTacToeGame


# Add different environments by adding a full capital key and a Game implementation class reference.
Games = {
    "HEX": HexGame,
    "GYM": GymGame,
    "ATARI": AtariGame,
    "IMAGEGYM": ImageGymGame,
    "TICTACTOE": TicTacToeGame
}
