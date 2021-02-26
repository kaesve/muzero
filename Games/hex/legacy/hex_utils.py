"""
Utility functions for the HexBoard class.
Includes functions for:
 - Finding all available moves on a HexBoard.
 - Finding all positions on a HexBoard where a player has placed pieces.
 - Checking validity of a move.
 - Performing moves on the HexBoard.
 - Removing pieces from the HexBoard.
 - Forcing pieces on one coordinate on the HexBoard.

EDIT: Modified the neccesary code to work with the new np.ndarray board representation

:version: FINAL
:date: 15-05-2020
:author: Joery de Vries and Ken Voskuil
:edited by: Joery de Vries and Ken Voskuil
"""

from Games.hex.HexLogic import HexBoard


def available_moves(hex_board):
    """
    Get all empty positions of the HexBoard = all available moves.
    :param hex_board: HexBoard class object
    :return: list of all empty positions on the current HexBoard.
    """
    return [(i, j) for i in range(hex_board.size) for j in range(hex_board.size) if hex_board.is_empty((i, j))]


def placed_positions(hex_board, color):
    """
    Retrieve all positions on the HexBoard where the given color has
    placed a piece.
    :param hex_board: HexBoard Class for game-logic
    :param color: int HexBoard player color
    :return: list of all positions on the board that belong to color.
    """
    return [(i, j) for i in range(hex_board.size)
            for j in range(hex_board.size)
            if hex_board.is_color((i, j), color)]


def is_valid_move(hex_board, coordinate):
    """
    Check if a given move is a valid move/ can be made.
    :param hex_board: HexBoard Class for game-logic.
    :param coordinate: tuple Coordinate on the HexBoard.
    :param color: int HexBoard player color.
    :return: boolean True if the move can be performed.
    """
    return all([0 <= c < hex_board.size for c in coordinate]) and hex_board.is_empty(coordinate)


def no_moves(hex_board):
    """
    Check whether there are moves left on the board.
    If the HexBoard is completely filled and no 'game-over' is indicated
    then the players have 'tied'.
    :param hex_board: HexBoard Class for game-logic
    :return: boolean True if there are no more possible moves on the hex_board
    :see: available_moves
    """
    return len(available_moves(hex_board)) == 0


def make_move(hex_board, coordinate, color):
    """
    Perform a move. Note: Does no checking for color or validity, see `valid_move'.
    :param hex_board: HexBoard Class for game-logic.
    :param coordinate: tuple Coordinate on the HexBoard.
    :param color: int HexBoard player color.
    :see: is_valid_move
    """
    hex_board.place(coordinate, color)


def unmake_move(hex_board, coordinate):
    """
    Unmake a previously made move by emptying the given position.
    :param hex_board: HexBoard Class for game-logic.
    :param coordinate: tuple Coordinate on the HexBoard.
    """
    hex_board.board[coordinate] = HexBoard.EMPTY
    hex_board.game_over = False


def emplace(hex_board, coordinate, color):
    """
    Force a piece at the coordinate on the board.
    :param hex_board: HexBoard Class for game-logic.
    :param coordinate: tuple Coordinate on the HexBoard.
    :param color: int HexBoard player color.
    """
    hex_board.board[coordinate] = color


def clear(hex_board):
    """
    Set all board positions to empty in the given HexBoard
    :param hex_board: HexBoard Class for game-logic.
    """
    for i in range(hex_board.size):
        for j in range(hex_board.size):
            hex_board.board[(i, j)] = hex_board.EMPTY
