from __future__ import print_function
import sys
import typing
from copy import deepcopy

import numpy as np

from Game import Game
import gym


class GymState:
    def __init__(self, env, observation: np.ndarray, action: int, done: bool):
        self.env = env
        self.observation = observation
        self.action = action
        self.done = done


sys.path.append('../../..')


class GymGame(Game):

    def __init__(self, env_name):
        super().__init__(n_players=1)
        self.env_name = env_name

        dummy = gym.make(env_name)
        self.dimensions = dummy.observation_space.shape
        self.actions = dummy.action_space.n

    def getDimensions(self, **kwargs) -> typing.Tuple[int, ...]:
        return self.dimensions if len(self.dimensions) > 1 else (1, 1, *self.dimensions)

    def getActionSize(self) -> int:
        return self.actions

    def getInitialState(self) -> np.ndarray:
        env = gym.make(self.env_name)
        observation = env.reset() if len(self.dimensions) > 1 else np.asarray([[env.reset()]])
        return GymState(env, observation, -1, False)

    def getNextState(self, state: GymState, action: int, player: int, **kwargs) -> typing.Tuple[GymState, float, int]:
        # Gym may raise warnings that .step() is called even though the environment is done.
        # This however doesn't happen and may be a results of DeepCopy in the AlphaMCTS procedure.
        observation, reward, done, info = state.env.step(action)
        observation = observation if len(self.dimensions) > 1 else [[observation]]

        def nextEnv(old_state, clone: bool = False):  # Macro for cloning the state
            return deepcopy(old_state) if clone else old_state

        env = nextEnv(state.env, **kwargs)
        return GymState(env, observation, action, done), reward, player

    def getLegalMoves(self, state: np.ndarray, player: int,
                      form: Game.Representation = Game.Representation.CANONICAL) -> np.ndarray:
        return np.ones(self.actions)

    def getGameEnded(self, state: GymState, player: int, close: bool = False) -> typing.Union[float, int]:
        if state.done and close:
            state.env.close()

        return int(state.done)

    def getSymmetries(self, state: np.ndarray, pi: np.ndarray, **kwargs) -> typing.List:
        return [(state, pi)]

    def getCanonicalForm(self, state: GymState, player: int) -> np.ndarray:
        return state

    def buildObservation(self, state: GymState, player: int, **kwargs) -> np.ndarray:
        return np.array(state.observation)

    def getHash(self, state: GymState) -> str:
        return np.asarray(state.observation).tobytes()
