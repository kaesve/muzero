"""
This python file wraps our implementation of hex with the backend of AlphaZero General.

See the report for the details on symmetry and how moves are correctly returned to
the player.

For documentation we refer to the game-logic class and the parent-class:
:see: HexLogic.py
:see: Game.py
"""

from __future__ import print_function
import sys
import typing

import numpy as np

from Games.Game import Game
from .HexLogic import HexBoard
from utils.game_utils import GameState

sys.path.append('../../..')


class HexGame(Game):
    square_content = {
        HexBoard.RED: "r",
        HexBoard.EMPTY: "-",
        HexBoard.BLUE: "b"
    }

    def __init__(self, n: int) -> None:
        super().__init__(n_players=2)
        self.n = n
        self.n_symmetries = 2

    def getInitialState(self) -> GameState:
        b = HexBoard(self.n)
        next_state = GameState(canonical_state=b.board, observation=None, action=-1, player=1, done=False)
        next_state.observation = self.buildObservation(next_state)
        return next_state

    def getDimensions(self) -> typing.Tuple[int, int, int]:
        # (a,b) tuple
        return self.n, self.n, 3

    def getActionSize(self) -> int:
        # return number of actions
        return self.n * self.n + 1

    def getNextState(self, state: GameState, action: int, **kwargs) -> typing.Tuple[GameState, float]:
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            state.done = True
            state.player = -state.player
            return state, 0

        b = HexBoard(self.n)
        b.board = np.copy(state.canonical_state)

        move = (action // self.n, action % self.n)
        assert b.board[move] == HexBoard.EMPTY

        # Perform action.
        b.place(move, state.player)

        next_state = GameState(canonical_state=b.board, observation=None, action=action,
                               player=-state.player, done=False)
        next_state.observation = self.buildObservation(state)

        return next_state, 0

    def getLegalMoves(self, state: GameState) -> np.ndarray:
        # return a fixed size binary vector
        b = HexBoard(self.n)
        b.board = np.copy(state.canonical_state)

        valid_moves = np.append(1 - np.abs(b.board.ravel()), 0)

        if np.sum(valid_moves) == 0:  # or self.getGameEnded(board, player) != 0:
            valid_moves[-1] = 1
            return valid_moves

        assert np.all([np.array(valid_moves[:-1]).reshape((self.n, self.n)) + np.abs(b.board) == 1])

        return valid_moves

    def getGameEnded(self, state: GameState, **kwargs) -> int:
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = HexBoard(self.n)
        b.board = np.copy(state.canonical_state)

        if b.check_win(1):
            state.done = True
            return 1 if state.player == 1 else -1

        if b.check_win(-1):
            state.done = True
            return 1 if state.player == -1 else -1

        return 0

    def buildObservation(self, state: GameState) -> np.ndarray:
        # Observation consists of three planes concatenated along the last dimension.
        # The first plane is an elementwise indicator for the board where player 1 is.
        # The second plane is identical to the first plane, but for player -1.
        # The third plane is a bias plane indicating whose turn it is (p1=1, p2=-1).
        # Output shape = (board_x, board_y, 3)
        board = state.canonical_state
        s_p1 = np.where(board == 1, 1.0, 0.0)
        s_p2 = np.where(board == -1, 1.0, 0.0)
        to_play = np.full_like(board, state.player)
        return np.stack([s_p1, s_p2, to_play], axis=-1)

    def getSymmetries(self, board: GameState, pi: np.ndarray) -> typing.List:
        # mirror, rotational
        assert (len(pi) == self.n ** 2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        symmetries = []

        # Only symmetry that Hex has is a 180 degree rotation.
        symmetries += [(board, pi)]
        symmetries += [(np.rot90(board, 2), list(np.rot90(pi_board, 2).ravel()) + [pi[-1]])]

        return symmetries

    def getHash(self, state: GameState) -> bytes:
        return state.canonical_state.tobytes()

    def stringRepresentationReadable(self, state: GameState):
        board_s = "".join(self.square_content[square] for row in state.canonical_state for square in row)
        return board_s

    def getScore(self, state: GameState):
        return len(HexBoard(state.canonical_state).get_empty_coordinates())

    @staticmethod
    def display(state: GameState):
        board_cls = HexBoard(state.canonical_state.shape[0])
        board_cls.board = state.canonical_state
        board_cls.print()
