from abc import ABC, abstractmethod
import typing

import numpy as np

from utils.game_utils import GameState


class Game(ABC):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    single-player, two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2 (if |players| > 1).

    See hex/HexGame.py for an example implementation.
    """

    def __init__(self, n_players: int = 1) -> None:
        self.n_players = n_players
        self.n_symmetries = 1

    @abstractmethod
    def getInitialState(self) -> GameState:
        """
        Returns:
            startState: a representation of the initial state (ideally this is the form
                        that will be the input to your neural network)
        """

    @abstractmethod
    def getDimensions(self) -> typing.Tuple[int, ...]:
        """
        Returns:
            (x,y): a tuple of the state dimensions
        """

    @abstractmethod
    def getActionSize(self) -> int:
        """
        Returns:
            actionSize: number of all possible actions
        """

    @abstractmethod
    def getNextState(self, state: GameState, action: int, **kwargs) -> typing.Tuple[GameState, float]:
        """
        Input:
            state: current state
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextState: State after applying action
            reward: Immediate observed reward (default should be 0 for most boardgames)
        """

    @abstractmethod
    def getLegalMoves(self, state: GameState) -> np.ndarray:
        """
        Input:
            board: current state
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are legal 0 for invalid moves
        """

    @abstractmethod
    def getGameEnded(self, state: GameState) -> typing.Union[float, int]:
        """
        Input:
            state: current state
            player: current player (1 or -1)

        Returns:
            z: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """

    @abstractmethod
    def buildObservation(self, state: GameState) -> np.ndarray:
        """
        Input:
            state: current state
            player: current player (1 or -1)
            form: Enum specification for how the observation should be constructed

        Returns:
            observation: Game specific implementation for what the neural network observes
                         at the state provided as an argument.
        """

    @abstractmethod
    def getSymmetries(self, state: GameState, pi: np.ndarray) -> typing.List:
        """
        Input:
            state: current state
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(state,pi)] where each tuple is a symmetrical
                       form of the state and the corresponding pi vector. This
                       is used when training AlphaZero from examples.
        """

    @abstractmethod
    def getHash(self, state: GameState) -> str:
        """
        Input:
            state: current state

        Returns:
            stateString: a quick conversion of state to a string format. Required by MCTS for hashing.
        """
