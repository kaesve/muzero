"""
This python file wraps our implementation of TicTacToe with the backend of AlphaZero General.

See the report for the details on symmetry and how moves are correctly returned to
the player.

For documentation we refer to the game-logic class and the parent-class:
:see: TicTactoeLogic.py
:see: Game.py
"""

from __future__ import print_function
import sys
import typing

import numpy as np

from Games.Game import Game
from .TicTacToeLogic import TicTacToeBoard
from utils.game_utils import GameState

sys.path.append('../../..')


class TicTacToeGame(Game):
    square_content = {
        TicTacToeBoard.X: "X",
        TicTacToeBoard.EMPTY: "-",
        TicTacToeBoard.O: "O"
    }

    def __init__(self, n: int) -> None:
        super().__init__(n_players=2)
        self.n = n
        self.n_symmetries = 2

    def getInitialState(self) -> GameState:
        b = TicTacToeBoard(self.n)
        next_state = GameState(canonical_state=b.board, observation=None, action=-1, player=1, done=False)
        next_state.observation = self.buildObservation(next_state)
        return next_state

    def getDimensions(self) -> typing.Tuple[int, int, int]:
        # (a,b) tuple
        return self.n, self.n, 3

    def getActionSize(self) -> int:
        # return number of actions
        # there is one action for each position on the board, and one action to concede
        return self.n * self.n + 1

    def getNextState(self, state: GameState, action: int, **kwargs) -> typing.Tuple[GameState, float]:
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n: # concede
            state.done = True
            state.player = -state.player
            return state, -1

        b = TicTacToeBoard(self.n)
        b.board = np.copy(state.canonical_state)

        move = (action // self.n, action % self.n)
        assert b.board[move] == TicTacToeBoard.EMPTY

        # Perform action.
        b.place(move, state.player)

        next_state = GameState(canonical_state=b.board, observation=None, action=action,
                               player=-state.player, done=False)

        z = -self.getGameEnded(next_state)  # Negated as this function is called from the adversary's perspective.
        next_state.observation = self.buildObservation(state)
        next_state.done = bool(z)

        return next_state, z

    def getLegalMoves(self, state: GameState) -> np.ndarray:
        # return a fixed size binary vector
        b = TicTacToeBoard(self.n)
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
        b = TicTacToeBoard(self.n)
        b.board = np.copy(state.canonical_state)

        winner = b.check_win()
        if winner == 0:
            return 0
        else:
            return 1 if state.player == winner else -1

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

        symmetries = [
            (board, pi),
            
            # Mirrors
            (np.flip(board, 0), list(np.flip(pi_board, 0).ravel()) + [pi[-1]]),
            (np.flip(board, 1), list(np.flip(pi_board, 1).ravel()) + [pi[-1]]),
            
            # Rotational symmetries
            *[ (np.rot90(board, r), list(np.rot90(pi_board, r).ravel()) + [pi[-1]]) for r in range(4) ],
        ]

        return symmetries

    def getHash(self, state: GameState) -> bytes:
        return state.canonical_state.tobytes()

    def stringRepresentationReadable(self, state: GameState):
        board_s = "".join(self.square_content[square] for row in state.canonical_state for square in row)
        return board_s

    def getScore(self, state: GameState):
        return len(TicTacToeBoard(state.canonical_state).get_empty_coordinates())

    def render(self, state: GameState):
        board_cls = TicTacToeBoard(state.canonical_state.shape[0])
        board_cls.board = state.canonical_state
        board_cls.print()
