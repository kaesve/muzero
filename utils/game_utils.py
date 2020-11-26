from dataclasses import dataclass
import typing

import gym
from gym import Env, spaces
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


class DiscretizeAction(gym.ActionWrapper):
    r"""Discretize the continuous action space of the environment into n steps.
    """
    def __init__(self, env, n):
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

    def action(self, action):
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low)*action/(self.action_space.n - 1)
        return np.array([ action ])