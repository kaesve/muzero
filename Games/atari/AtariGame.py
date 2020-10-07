from __future__ import print_function
import sys

from Game import Game
import numpy as np

import gym


sys.path.append('../../..')


class GymState:
    def __init__(self, env, observation, done):
        self.env = env
        self.observation = observation
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
        #TODO: check if this name is valid?
        super().__init__()
        self.env_name = env_name

    def getInitialState(self):
        """
        Returns:
            startState: a representation of the initial state (ideally this is the form
                        that will be the input to your neural network)
        """
        env = gym.make(env_name)
        return GymState(env, env.reset(), False)

    def getDimensions(self):
        """
        Returns:
            (x,y): a tuple of the state dimensions
        """

        return (210, 160, 3)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 6

    def getNextState(self, state, action, player):
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

        observation, reward, done, info = state.env.step(action)
        return (GymState(env, observation, done), reward, 1)

    def getLegalMoves(self, state, player):
        """
        Input:
            board: current state
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are legal 0 for invalid moves
        """
        return np.identity(self.getActionSize())

    def getGameEnded(self, state, player):
        """
        Input:
            state: current state
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        return state.done

    def getCanonicalForm(self, state, player):
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

    def buildTrajectory(self, history, current_state, current_player, t=None):
        """
        Input:
            history: Some data structure that can be queried for past observations.
            current_state: Current observation of the environment.
            current_player: Current player (-1 or 1)
            t: optional Specifies the index of current state within history (if present)

        Returns:
            trajectory: Game specific array of observed features over time.
                        Feature planes are flattened over time, such that
                        the number of planes = time x features.
        """
        pass

    def getSymmetries(self, state, pi):
        """
        Input:
            state: current state
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(state,pi)] where each tuple is a symmetrical
                       form of the state and the corresponding pi vector. This
                       is used when training AlphaZero from examples.
        """
        pass

    def stringRepresentation(self, state):
        """
        Input:
            state: current state

        Returns:
            stateString: a quick conversion of state to a string format.
                         Required by MCTS for hashing.
        """
        pass

    def buildTrajectory(self, history):
        pass
