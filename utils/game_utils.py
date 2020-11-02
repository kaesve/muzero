from dataclasses import dataclass
import typing

from gym import Env
from gym.envs.atari import AtariEnv
import numpy as np


@dataclass
class GameState:
    canonical_state: typing.Any  # s_t
    observation: np.ndarray      # o_t
    action: int                  # a_t
    player: int                  # player_t
    done: bool                   # I(s_t = s_T)


@dataclass
class GymState(GameState):
    env: Env  # Class for the (stateful) logic of Gym Environments at t.


@dataclass
class AtariState(GymState):
    env: AtariEnv  # Class for the (stateful) logic of Gym Atari Environments at t.
