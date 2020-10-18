from __future__ import print_function
import sys
import typing

from Game import Game
import numpy as np

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

    def getDimensions(self, form: Game.Representation = Game.Representation.CANONICAL) -> typing.Tuple[int, ...]:
        return self.dimensions

    def getActionSize(self) -> int:
        return self.actions

    def getInitialState(self) -> np.ndarray:
        env = gym.make(self.env_name)
        return GymState(env, env.reset(), -1, False)

    def getNextState(self, state: GymState, action: int, player: int,
                     form: Game.Representation = Game.Representation.CANONICAL) -> typing.Tuple[GymState, float, int]:
        observation, reward, done, info = state.env.step(action)
        return GymState(state.env, observation, action, done), reward, player

    def getLegalMoves(self, state: np.ndarray, player: int,
                      form: Game.Representation = Game.Representation.CANONICAL) -> np.ndarray:
        return np.ones(self.actions)

    def getGameEnded(self, state: GymState, player: int) -> typing.Union[float, int]:
        if state.done:
            state.env.close()

        return int(state.done)

    def getCanonicalForm(self, state: GymState, player: int) -> np.ndarray:
        return state

    def buildObservation(self, state: GymState, player: int,
                         form: Game.Representation = Game.Representation.CANONICAL) -> np.ndarray:
        return np.array(state.observation)
