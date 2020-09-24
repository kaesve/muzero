"""
This python file wraps our implementation of hex with the backend of AlphaZero General.

See the report for the details on symmetry and how moves are correctly returned to
the player.

For documentation we refer to the game-logic class and the parent-class:
:see: hex_skeleton.py
:see: Game.py
"""

from __future__ import print_function
import sys
import numpy as np

from Game import Game
from .src_joery.hex_skeleton import HexBoard
from .src_joery.hex_utils import available_moves, make_move

sys.path.append('..')


class HexGame(Game):
    square_content = {
        HexBoard.RED: "r",
        HexBoard.EMPTY: "-",
        HexBoard.BLUE: "b"
    }

    def __init__(self, n):
        super().__init__()
        self.n = n

    def getInitBoard(self):
        super().getInitBoard()
        # return initial board (numpy board)
        b = HexBoard(self.n)
        return b.board

    def getDimensions(self):
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        # return number of actions
        return self.n * self.n + 1

    def getNextState(self, state, player, action):
        super().getNextState(state, player, action)
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return state, -player
        b = HexBoard(self.n)
        b.board = np.copy(state)

        move = (action // self.n, action % self.n)
        if player == -1:  # Make the move on the transposed board
            move = move[::-1]

        assert b.board[move] == HexBoard.EMPTY

        make_move(b, move, player)

        return b.board, -player

    def getLegalMoves(self, state, player):
        # return a fixed size binary vector
        b = HexBoard(self.n)
        b.board = np.copy(state)

        # Order of raveling is done in normal order C or in tranposed order F depending on the player.
        valids = np.append(1 - np.abs(b.board.ravel(order='C' if player == 1 else 'F')), 0)
        if np.sum(valids) == 0:  # or self.getGameEnded(board, player) != 0:
            valids[-1] = 1
            return np.array(valids)

        assert np.all(np.array(valids[:-1]).reshape((self.n, self.n), order='C') + np.abs(b.board) == 1)

        return valids

    def getGameEnded(self, state, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = HexBoard(self.n)
        b.board = np.copy(state)

        if b.check_win(1):
            return 1 if player == 1 else -1

        if b.check_win(-1):
            return 1 if player == -1 else -1

        return 0

    def getCanonicalForm(self, state, player):
        return state if player == 1 else -state.T

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert (len(pi) == self.n ** 2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        syms = []

        # Only symmetry that Hex has is a 180 degree rotation.
        syms += [(board, pi)]
        syms += [(np.rot90(board, 2), list(np.rot90(pi_board, 2).ravel()) + [pi[-1]])]

        # Optionally, the board's perspective can be flipped by:
        # -board.T
        # However, this is not as simple for the policy, pi. As pi was generated in terms
        # for the adversary. Not correctly modifying the policy would imply learning the
        # adversary's policy for the player's perspective (i.e., learn to lose). Hence,
        # we did not use this additional symmetry.

        return syms

    def stringRepresentation(self, state):
        return state.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        return len(available_moves(board))

    @staticmethod
    def display(board):
        board_cls = HexBoard(board.shape[0])
        board_cls.board = board
        board_cls.print()
