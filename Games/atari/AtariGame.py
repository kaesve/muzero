from __future__ import print_function
import typing
import sys

import numpy as np
import gym

from Games.gym.GymGame import GymGame
from utils.game_utils import AtariState

sys.path.append('../../..')


class AtariGame(GymGame):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    single-player, two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2 (if |players| > 1).

    See hex/HexGame.py for an example implementation.
    """

    def __init__(self, env_name: str) -> None:
        super().__init__(env_name)

    def getInitialState(self) -> AtariState:
        """
        Returns:
            startState: a representation of the initial state (ideally this is the form
                        that will be the input to your neural network)
        """
        env = gym.make(self.env_name)
        env = gym.wrappers.AtariPreprocessing(env, screen_size=96, scale_obs=True, grayscale_obs=False,
                                              terminal_on_life_loss=True, noop_max=10)
        root = env.reset()

        if 'FIRE' in env.unwrapped.get_action_meanings():
            # The wrapper from the baseline repo implies so:
            # https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
            root, *_ = env.step(1)

        next_state = AtariState(canonical_state=root, observation=None,
                                action=-1, done=False, player=1, env=env)
        next_state.observation = self.buildObservation(next_state)

        return next_state

    def getDimensions(self, **kwargs) -> typing.Tuple[int, int, int]:
        """
        Returns:
            (x,y,z): a tuple of the state dimensions
        """

        return 96, 96, 4

    def getActionSize(self) -> int:
        """ Return the number of possible actions """
        # The latent space is hard coded to be 6x6 = 36
        return 36

    def getNextState(self, state: AtariState, action: int, **kwargs) -> typing.Tuple[AtariState, float]:
        """
        Input:
            state: current state
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextState: State after applying action
            reward: Immediate observed reward (default should be 0 for most boardgames)
            nextPlayer: player who plays in the next turn
        """
        def nextEnv(old_state: AtariState, clone: bool = False):  # Macro for cloning the state
            return old_state.env.clone_full_state() if clone else old_state.env

        env = nextEnv(state, **kwargs)

        # reorder actions so that we can have NOOP at the end
        action = (action - 1 + self.actions) % self.actions
        raw_observation, reward, done, info = env.step(action)

        next_state = AtariState(canonical_state=raw_observation, observation=None,
                                action=action, done=done, player=1, env=env)
        next_state.observation = self.buildObservation(next_state)

        return next_state, reward

    def getLegalMoves(self, state: AtariState, **kwargs) -> np.ndarray:
        """
        Input:
            board: current state
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are legal 0 for invalid moves
        """
        res = np.arange(self.getActionSize()) < self.actions
        return res.astype(int)

    def buildObservation(self, state: AtariState, **kwargs) -> np.ndarray:
        action_plane = np.ones(state.canonical_state.shape[:2]) * state.action / self.getActionSize()
        action_plane = action_plane.reshape((*action_plane.shape, 1))
        return np.concatenate((state.canonical_state, action_plane), axis=-1)

    def render(self, state: AtariState):
        state.env.render()
