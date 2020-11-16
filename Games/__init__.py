"""
Initialization module to define the available Game implementations in the 'Games' scope.
"""
from .atari.AtariGame import AtariGame
from .gym.GymGame import GymGame
from .hex.HexGame import HexGame


# Add different environments by adding a full capital key and a Game implementation class reference.
Games = {
    "HEX": HexGame,
    "GYM": GymGame,
    "ATARI": AtariGame
}
