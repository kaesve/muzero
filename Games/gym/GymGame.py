from __future__ import print_function
import sys
import typing
from copy import deepcopy

import numpy as np
import gym

from Games.Game import Game
from utils.game_utils import GymState

sys.path.append('../../..')


class GymGame(Game):

    def __init__(self, env_name: str) -> None:
        super().__init__(n_players=1)
        self.env_name = env_name

        dummy = gym.make(env_name)
        self.dimensions = dummy.observation_space.shape
        self.actions = dummy.action_space.n

    def getDimensions(self, **kwargs) -> typing.Tuple[int, ...]:
        return self.dimensions if len(self.dimensions) > 1 else (1, 1, *self.dimensions)

    def getActionSize(self) -> int:
        return self.actions

    def getInitialState(self) -> GymState:
        env = gym.make(self.env_name)

        next_state = GymState(canonical_state=env.reset(), observation=None, action=-1, done=False, player=1, env=env)
        next_state.observation = self.buildObservation(next_state)

        return next_state

    def getNextState(self, state: GymState, action: int, **kwargs) -> typing.Tuple[GymState, float]:
        def nextEnv(old_state: GymState, clone: bool = False):  # Macro for cloning the state
            return deepcopy(old_state.env) if clone else old_state.env

        env = nextEnv(state, **kwargs)
        raw_observation, reward, done, info = env.step(action)

        next_state = GymState(canonical_state=raw_observation, observation=None,
                              action=action, done=done, player=1, env=env)
        next_state.observation = self.buildObservation(next_state)

        return next_state, reward

    def getLegalMoves(self, state: GymState) -> np.ndarray:
        return np.ones(self.actions)

    def getGameEnded(self, state: GymState, **kwargs) -> int:
        return int(state.done)

    def getSymmetries(self, state: GymState, pi: np.ndarray, **kwargs) -> typing.List:
        return [(state.observation, pi)]

    def buildObservation(self, state: GymState, **kwargs) -> np.ndarray:
        return state.canonical_state if len(self.dimensions) > 1 else np.asarray([[state.canonical_state]])

    def getHash(self, state: GymState) -> bytes:
        return np.asarray(state.canonical_state).tobytes()

    def close(self, state: GymState) -> None:
        state.env.close()

    def render(self, state: GymState):
        state.env.render()
