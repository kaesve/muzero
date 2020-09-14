"""
Sample Python debugging file for testing out functions from this
package. For example testing of Heuristic and board-logic functions.

:version: FINAL
:date: 07-02-2020
:author: Aske Plaat
:edited by: Joery de Vries
"""

from Assignment_1.src import *

# sanity check that wins are detected
for i in range(0, 2):
    winner = HexBoard.RED if i == 0 else HexBoard.BLUE
    looser = HexBoard.BLUE if i == 0 else HexBoard.RED
    board = HexBoard(3)
    board.place((1, 1), looser)
    board.place((2, 1), looser)
    board.place((1, 2), looser)
    board.place((2, 2), looser)
    board.place((0, 0), winner)
    board.place((1, 0), winner)
    board.place((2, 0), winner)
    board.place((0, 1), winner)
    board.place((0, 2), winner)
    assert (board.check_win(winner) == True)
    assert (board.check_win(looser) == False)
    board.print()

endable_board = HexBoard(4)
# sanity check that random play will at some point end the game
while not endable_board.game_over:
    endable_board.place((np.random.randint(0, 4), np.random.randint(0, 4)), HexBoard.RED)
assert (endable_board.game_over == True)
assert (endable_board.check_win(HexBoard.RED) == True)
assert (endable_board.check_win(HexBoard.BLUE) == False)
print("Randomly filled board")
endable_board.print()

neighbor_check = HexBoard(5)
assert (neighbor_check.get_neighbors((0, 0)) == [(1, 0), (0, 1)])
assert (neighbor_check.get_neighbors((0, 1)) == [(1, 1), (1, 0), (0, 2), (0, 0)])
assert (neighbor_check.get_neighbors((1, 1)) == [(0, 1), (2, 1), (0, 2), (2, 0), (1, 2), (1, 0)])
assert (neighbor_check.get_neighbors((3, 4)) == [(2, 4), (4, 4), (4, 3), (3, 3)])
assert (neighbor_check.get_neighbors((4, 3)) == [(3, 3), (3, 4), (4, 4), (4, 2)])
assert (neighbor_check.get_neighbors((4, 4)) == [(3, 4), (4, 3)])

neighbor_check_small = HexBoard(2)
assert (neighbor_check_small.get_neighbors((0, 0)) == [(1, 0), (0, 1)])
assert (neighbor_check_small.get_neighbors((1, 0)) == [(0, 0), (0, 1), (1, 1)])
assert (neighbor_check_small.get_neighbors((0, 1)) == [(1, 1), (1, 0), (0, 0)])
assert (neighbor_check_small.get_neighbors((1, 1)) == [(0, 1), (1, 0)])

neighbor_check_sanity = HexBoard(11)
for x in range(0, 11):
    for y in range(0, 11):
        neighbors = neighbor_check_sanity.get_neighbors((x, y))
        for neighbor in neighbors:
            neighbors_neighbors = neighbor_check_sanity.get_neighbors(neighbor)
            index_of_self = neighbors_neighbors.index((x, y))
            assert (index_of_self != -1)


board = HexBoard(4)
moves = available_moves(board)
assert len(moves) == 4**2

make_move(board, moves[0], HexBoard.RED)
new_moves = available_moves(board)
assert len(new_moves) == 4**2-1

unmake_move(board, moves[0])
moves = available_moves(board)
assert len(moves) == 4**2

# Hueristic testing
dijkstra = DijkstraHeuristic()
board.print()
cost = dijkstra._evaluate(board, HexBoard.RED)
print('cost:', cost)

make_move(board, (3,2), HexBoard.BLUE)
make_move(board, (2,2), HexBoard.BLUE)
make_move(board, (1,2), HexBoard.BLUE)
make_move(board, (0,2), HexBoard.BLUE)
board.print()
cost = dijkstra._evaluate(board, HexBoard.BLUE)
print('cost:', cost)
print(board.game_over)

if __name__ == "__main__":
    print("Tests done.")
