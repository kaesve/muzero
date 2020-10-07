"""
Function to play a simple hex-game between two provided
players on a game-board of a specified size.

:version: FINAL
:date: 07-02-2020
:author: Joery de Vries
:edited by: Joery de Vries, Oliver Konig, Siyuan Dong
"""

from .hex_skeleton import HexBoard
from .hex_utils import is_valid_move, make_move


def playgame(boardsize, red, blue, doubles=False, verbose=True):
    """
    Function to play a simple hex-game between two provided
    players on a game-board of a specified size.

    If double is True then red wil also play as blue. The winner will
    then be the player that wins as both red AND blue. Otherwise it is a draw.

    :param boardsize: int Size of the game-board to play on.
    :param red: Player object for player RED.
    :param blue: Player object for player BLUE.
    :param doubles: bool True to let red also play once as blue.
    :param verbose: boolean Whether to print out the board between moves.
    :return: int Returns HexBoard player color for the winner (EMPTY if draw in doubles)
    :see: .hex_player Player Class for a hex-player
    :see: .hex_skeleotn HexBoard Class for the hex-game logic
    """
    board = HexBoard(boardsize)

    player = red
    while not board.is_game_over():
        if verbose:
            print('\n\n')
            board.print()
            if player == red:
                print("Red to move...")
            else:
                print("Blue to move...")

        move = player.select_move(board)
        if not is_valid_move(board, move):  # non-valid move is seen as a 'pass'.
            break

        make_move(board, coordinate=move, color=player.color)
        player = blue if player == red else red

    # Red wins if player == blue
    result = HexBoard.RED if player == blue else HexBoard.BLUE

    if verbose:
        board.print()
        if player == blue:
            print("Red wins!")
        else:
            print("Blue wins!")

    if doubles:
        red.switch(HexBoard.BLUE)
        blue.switch(HexBoard.RED)
        # Score is negated as the players are switched!
        result_switch = playgame(boardsize, red=blue, blue=red, doubles=False, verbose=verbose)
        result = result if result == board.get_opposite_color(result_switch) else HexBoard.EMPTY

    return result
