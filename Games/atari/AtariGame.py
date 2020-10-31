from __future__ import print_function
import typing
import sys

from Game import Game
import numpy as np

import gym

sys.path.append('../../..')


class GymState:
    def __init__(self, env, observation, action, done):
        self.env = env
        self.observation = observation
        self.action = action
        self.done = done


class AtariGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    single-player, two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2 (if |players| > 1).

    See hex/HexGame.py for an example implementation.
    """

    def __init__(self, env_name):
        super().__init__()
        self.env_name = env_name
        self.action_size = gym.make(env_name).action_space.n

    def getInitialState(self):
        """
        Returns:
            startState: a representation of the initial state (ideally this is the form
                        that will be the input to your neural network)
        """
        env = gym.make(self.env_name)
        env = gym.wrappers.AtariPreprocessing(env, screen_size=96, scale_obs=True, grayscale_obs=False,
                                              terminal_on_life_loss=True,
                                              noop_max=30)
        return GymState(env, env.reset(), 0, False)

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

    def getNextState(self, state: GymState, action: int, player: int, **kwargs):
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
        def nextEnv(old_state, clone: bool = False):  # Macro for cloning the state
            return old_state.env.clone_full_state() if clone else old_state.env

        env = nextEnv(state, **kwargs)
        # reorder actions so that we can have NOOP at the end
        action = (action + 1) % self.action_size
        observation, reward, done, info = env.step(action)

        return GymState(env, observation, action, done), reward, 1

    def getLegalMoves(self, state: GymState, player: int, **kwargs):
        """
        Input:
            board: current state
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are legal 0 for invalid moves
        """
        res = np.arange(self.getActionSize()) < self.action_size
        res = res.astype(int)
        return res

    def getGameEnded(self, state: GymState, player: int, close: bool = False) -> int:
        """
        Input:
            state: current state
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """

        if state.done and close:
            state.env.close()

        return int(state.done)

    def getCanonicalForm(self, state: np.ndarray, player: int):
        """
        Input:
            state: current state
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of the state. The canonical form
                            should be independent of the player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return state

    def buildObservation(self, state: GymState, player: int,
                         form: Game.Representation = Game.Representation.HEURISTIC) -> np.ndarray:
        if form == Game.Representation.CANONICAL:
            return self.getCanonicalForm(state, player).observation

        elif form == Game.Representation.HEURISTIC:
            action_plane = np.ones(state.observation.shape[:2]) * state.action / self.action_size
            action_plane = action_plane.reshape((*action_plane.shape, 1))
            return np.concatenate((state.observation, action_plane), axis=2)

    def getSymmetries(self, state: GymState, pi: np.ndarray, **kwargs):
        return [(state.observation, pi)]

    def getHash(self, state: GymState):
        return state.observation.tobytes()
