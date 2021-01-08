"""
File to define utilities for Game handling. The GameState data structures serve as the states that preserve the
information of an environment and is used within the Coach classes to handle environment data.
"""
from dataclasses import dataclass
import typing

import gym
from gym import Env, spaces
from gym.envs.atari import AtariEnv
import numpy as np


@dataclass
class GameState:
    canonical_state: typing.Any     # s_t
    observation: np.ndarray         # o_t
    action: int     # a_t
    player: int     # player_t
    done: bool      # I(s_t = s_T)


@dataclass
class GymState(GameState):
    env: Env        # Class for the (stateful) logic of Gym Environments at t.


@dataclass
class AtariState(GymState):
    env: AtariEnv   # Class for the (stateful) logic of Gym Atari Environments at t.


class DiscretizeAction(gym.ActionWrapper):
    """
    Factorizes a continuous action space of an environment into n discrete actions.
    """

    def __init__(self, env, n: int) -> None:
        """
        Factorize the given environment's action space (a single continuous action) to n discrete actions.
        :param env: Gym.Env Environment object from OpenAI Gym.
        :param n: int Number of actions to factorize.
        """
        assert isinstance(env.action_space, spaces.Box), (
            "expected Box action space, got {}".format(type(env.action_space)))
        assert env.action_space.is_bounded(), "expected bounded Box action space"

        # We could support multiple dimensions, but that quickly becomes unmanagble with
        # the single dimension spaces.Discrete. We can add a version using
        # spaces.MultiDiscrete for that use case.
        dims = np.product(env.action_space.shape)
        assert dims == 1, f"expected 1d Box action space, got {dims}d space"

        super(DiscretizeAction, self).__init__(env)
        self.action_space = spaces.Discrete(n)

    def action(self, action: int) -> float:
        """
        Linearly scale the action integer between the continuous range.
        Example if range: [-1, 1] and n = 3, then a'=0 -> a=-1, a=1 -> a'=0, a=2 -> a'=1
        :param action: int Action bin to perform in the environment.
        :return: float Action cast to the original, continuous, action space.
        """
        low = self.env.action_space.low
        high = self.env.action_space.high

        action = low + (high - low) * action / (self.action_space.n - 1)

        return action

    def reverse_action(self, action: float) -> int:
        """ Yield the closest bin action to the given continuous action. TODO """
        pass
