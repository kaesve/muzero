from __future__ import print_function
import sys
import typing

import numpy as np

from Games.Game import Game
from .OthelloLogic import Board
from utils.game_utils import GameState

sys.path.append('../../..')


class OthelloGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    def __init__(self, n: int) -> None:
        super().__init__(n_players=2)
        self.n = n
        self.n_symmetries = 4

    def getInitialState(self):
        # return initial board (numpy board)
        b = Board(self.n)
        next_state = GameState(canonical_state=b.pieces, observation=None, action=-1, player=1, done=False)
        next_state.observation = self.buildObservation(next_state)
        return next_state

    def getDimensions(self, **kwargs) -> typing.Tuple[int, int, int]:
        # (a,b) tuple
        return self.n, self.n, 1

    def getActionSize(self) -> int:
        # return number of actions
        return self.n * self.n + 1

    def getNextState(self, state: GameState, action: int, **kwargs) -> typing.Tuple[GameState, float]:
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            state.player = -1
            state.done = True
            return state, 0

        b = Board(self.n)
        b.pieces = np.copy(state.canonical_state)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, state.player)

        next_state = GameState(canonical_state=b.pieces, observation=None, action=action,
                               player=-state.player, done=False)
        next_state.observation = self.buildObservation(next_state)
        return next_state, 0

    def getLegalMoves(self, state: GameState, **kwargs) -> np.ndarray:
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(state.canonical_state)
        legal_moves = b.get_legal_moves(state.player)

        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)

        for x, y in legal_moves:
            valids[self.n * x + y] = 1

        return np.array(valids)

    def getGameEnded(self, state: GameState, **kwargs) -> int:
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(state.observation)
        if b.has_legal_moves(state.player):
            return 0
        if b.has_legal_moves(-state.player):
            return 0

        state.done = True
        if b.countDiff(state.player) > 0:
            return 1
        return -1

    def buildObservation(self, state: GameState) -> np.ndarray:
        s_p1 = np.where(state == 1, 1.0, 0.0)
        s_p2 = np.where(state == -1, 1.0, 0.0)
        to_play = np.full_like(state, state.player)

        return np.stack([s_p1, s_p2, to_play], axis=-1)

    def getSymmetries(self, board: GameState, pi: np.ndarray, **kwargs) -> typing.List:
        # mirror, rotational
        assert len(pi) == self.n ** 2 + 1  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        syms = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board.observation, i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                syms += [(new_b, list(new_pi.ravel()) + [pi[-1]])]
        return syms

    def getHash(self, state: GameState) -> bytes:
        return state.canonical_state.tobytes()

    def stringRepresentationReadable(self, board: GameState) -> str:
        board_s = "".join(self.square_content[square] for row in board.canonical_state for square in row)
        return board_s

    def getScore(self, board: GameState, player: int) -> float:
        b = Board(self.n)
        b.pieces = np.copy(board.canonical_state)
        return b.countDiff(player)

    @staticmethod
    def display(board: GameState) -> None:
        b = board.canonical_state
        n = b.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = b[y][x]  # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
