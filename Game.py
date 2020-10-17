import typing
from enum import Enum

import numpy as np


class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    single-player, two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2 (if |players| > 1).

    See hex/HexGame.py for an example implementation.
    """

    class Representation(Enum):
        CANONICAL: int = 1
        HEURISTIC: int = 2

    def __init__(self, n_players: int = 1) -> None:
        self.n_players = n_players

    def getInitialState(self) -> np.ndarray:
        """
        Returns:
            startState: a representation of the initial state (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def getDimensions(self, form: Representation = Representation.CANONICAL) -> typing.Tuple[int, ...]:
        """
        Returns:
            (x,y): a tuple of the state dimensions
        """
        pass

    def getActionSize(self) -> int:
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def getNextState(self, state: np.ndarray, action: int, player: int,
                     form: Representation = Representation.CANONICAL) -> typing.Tuple[np.ndarray, float, int]:
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
        pass

    def getLegalMoves(self, state: np.ndarray, player: int, form: Representation = Representation.CANONICAL) -> np.ndarray:
        """
        Input:
            board: current state
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are legal 0 for invalid moves
        """
        pass

    def getGameEnded(self, state: np.ndarray, player: int) -> typing.Union[float, int]:
        """
        Input:
            state: current state
            player: current player (1 or -1)

        Returns:
            z: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

    def getCanonicalForm(self, state: np.ndarray, player: int) -> np.ndarray:
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
        pass

    def buildObservation(self, state: np.ndarray, player: int, form: Representation = Representation.CANONICAL) -> np.ndarray:
        """
        Input:
            state: current state
            player: current player (1 or -1)
            form: Enum specification for how the observation should be constructed

        Returns:
            observation: Game specific implementation for what the neural network observes
                         at the state provided as an argument.
        """
        pass

    def getSymmetries(self, state: np.ndarray, pi: np.ndarray,
                      form: Representation = Representation.CANONICAL) -> typing.List:
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

    def stringRepresentation(self, state: np.ndarray) -> str:
        """
        Input:
            state: current state

        Returns:
            stateString: a quick conversion of state to a string format.
                         Required by MCTS for hashing.
        """
        pass
