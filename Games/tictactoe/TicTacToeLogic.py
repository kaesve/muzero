"""
Class for the tic tac toe game logic. Based on the hex game logic code.

EDIT: Changed board representation from dict to numpy.ndarray.
EDIT: Changed numbers for the player-color indicators.

:version: FINAL
:date:
:author: Aske Plaat
:edited by: Joery de Vries
:edited by: Ken Voskuil
"""
import numpy as np


class TicTacToeBoard:
    O = -1  # 1
    X = 1  # 2
    EMPTY = 0  # 3

    def __init__(self, board_size):
        self.board = np.full((board_size, board_size), TicTacToeBoard.EMPTY)  # Used to be represented by a dict.
        self.size = board_size
        self.game_over = False

    def is_game_over(self):
        return self.game_over

    def is_empty(self, coordinates):
        return self.board[coordinates] == TicTacToeBoard.EMPTY

    def is_color(self, coordinates, color):
        return self.board[coordinates] == color

    def get_color(self, coordinates):
        if coordinates == (-1, -1):
            return TicTacToeBoard.EMPTY
        return self.board[coordinates]

    def place(self, coordinates, color):
        if not self.game_over and self.board[coordinates] == TicTacToeBoard.EMPTY:
            self.board[coordinates] = color
            self.game_over = self.check_win != TicTacToeBoard.EMPTY

    def get_opposite_color(self, current_color):
        if current_color == TicTacToeBoard.O:
            return TicTacToeBoard.X
        return TicTacToeBoard.O

    def get_empty_coordinates(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.is_empty((i, j))]

    def check_win(self):
        has_won = False
        cols = np.sum(self.board, axis=0)
        rows = np.sum(self.board, axis=1)
        diags = np.array([0, 0])

        for i in range(self.size):
            diags += [self.board[i, i], self.board[-i - 1, i]]

        concat = [*cols, *rows, *diags]
        if np.min(concat) == -self.size:
            return TicTacToeBoard.O
        elif np.max(concat) == self.size:
            return TicTacToeBoard.X
        else:
            return TicTacToeBoard.EMPTY

    def print(self):
        print("   ", end="")
        for y in range(self.size):
            print(chr(y + ord('a')), "", end="")
        print("")
        print("  --" + "--" * self.size + "-")
        for y in range(self.size):
            print(y, "| ", end="")
            for x in range(self.size):
                piece = self.board[x, y]
                if piece == TicTacToeBoard.O:
                    print("O ", end="")
                elif piece == TicTacToeBoard.X:
                    print("X ", end="")
                else:
                    if x == self.size:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")
        print("  --" + "--" * self.size + "-")
