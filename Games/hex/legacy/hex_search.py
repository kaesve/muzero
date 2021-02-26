"""
This Python file defines the Classes for search engines that search for
moves for playing the game of hex. Currently Minimax by the
Negamax alpha-beta method is implemented along with Monte-Carlo Tree Search.

:version: FINAL
:date: 07-02-2020
:author: Joery de Vries and Ken Voskuil
:edited by: Joery de Vries and Ken Voskuil

:bibliography:
1. Stuart Russell and Peter Norvig. 2009. Artificial Intelligence: A Modern
   Approach (3rd. ed.). Prentice Hall Press, USA. http://aima.cs.berkeley.edu/
"""

from .hex_utils import available_moves, make_move, unmake_move, no_moves, emplace

import time

import numpy as np


class Searcher(object):
    """
    Abstract class to provide a format for an environment state searcher.

    As the search can be time-bounded the class must store its best current
    move in a class variable, retrievable by get_move. If the search is
    terminated mid-search, then there will always be an available move.
    """

    def __init__(self):
        self.move = None

    def get_move(self):
        return self.move


class MinimaxSearcher(Searcher):
    """
    Derived class of Searcher which implements the Minimax algorithm with the
    Negamax alpha-beta method. This class implements:
     - Transposition table functionality
     - Iterative deepening
     - Fixed-depth search
     - Move-ordering
     - Budgeting of the heuristic evaluation function calls.


    """
    TERMINAL = (-1, -1)
    _VALUE_IDX = 0
    _MOVE_IDX = 1
    _EVAL_DEPTH_IDX = 2

    def __init__(self, player, heuristic):
        """
        Initalize the search engine with a player perspective (ally) along with
        a heuristic scoring functions for board states.
        :param player: int HexBoard player color.
        :param heuristic: HexHeuristic Derived class that implements a scoring function.
        :see: HexHeuristic in .hex_heuristics
        """
        super().__init__()
        self.evaluations = self.budget = 0
        self.itd_depth = 1
        self.player = player
        self.heuristic = heuristic
        self.move = self.TERMINAL
        self.hashseed = None
        self.transpositions = [{}]

    @property
    def use_hashing(self):
        """
        If the transposition table hashseed is not initialized it is implied
        that no hashing of states is used.
        :return: boolean True if it is implied that transpositions are used.
        """
        return self.hashseed is not None

    @property
    def _undefined_node(self):
        """
        Undefined node property. Search defaults to this node when resources run out.
        :return: (int, tuple, int) Heuristic value, Coordinate, and Search depth.
        """
        return -np.inf, self.TERMINAL, np.inf

    @property
    def hashtable(self):
        """
        Retrieve the transposition table at the CURRENT search iteration.
        With iterative deepening the algorithm uses transpositions of PREVIOUS
        iterations for move ordering, and not for retrieving values; this would
        cause the algorithm to get stuck at the first layer.
        :return: int Heuristic table at the current search iteration.
        """
        return self.transpositions[self.itd_depth - 1]

    def evaluate(self, hex_board, player):
        """
        Call the heuristic scoring function to evaluate the beneficiality of the
        current board-state for the given player.
        (Also increments the evaluation function call counter.)
        :param hex_board: HexBoard Class for game-logic.
        :param player: int HexBoard player color.
        :return: int Heuristic score of the provided board state for player.
        """
        self.evaluations += 1
        return self.heuristic.evaluate(hex_board, player)

    def initalize_transposition(self, size):
        """
        (Re-)Initialize the transposition table and generate a hashseed fitting matching
        the board's size.
        The hashseed contains random integers in a matrix of dimensionality size^2 x 2.
        Every position (flattened into a 1-d array) for every player then has its own seed.
        :param size: int Size of the game-board.
        """
        self.transpositions = [dict()]  # Empty transposition table to remove earlier moves
        self.hashseed = np.random.randint(0, 2 ** 31, size * size * 2).reshape((2, size * size))

    def zobrist(self, hex_board):
        """
        Compute a hash-value for a given board state. The hashing is performed by consequently
        XOR-ing the hasheeds of each board-position -- which is deterministic, and usually unique
        for most board-states. I.e., zobrist: U -> Z  maps a universe to distinct integers.
        :param hex_board: HexBoard Class for game-logic.
        :return: int Hashed value of the HexBoard.
        :references: https://en.wikipedia.org/wiki/Zobrist_hashing
        """
        board_hash = 0
        for pos, value in np.ndenumerate(hex_board.board):  # format= (row, col): value
            if not hex_board.is_empty(pos):
                # Select the indices of the hashseed corresponding to the current coordinate.
                type_index = 0 if hex_board.is_color(pos, self.player) else 1
                linear_hash_index = pos[0] * hex_board.size + pos[1]

                # Subsequently keep XOR-ing the hash-value with the selected hashseed.
                board_hash = np.bitwise_xor(board_hash, self.hashseed[type_index, linear_hash_index])

        return board_hash

    def order_moves(self, hex_board, moves, player):
        """
        Orders the available moves at the current state of the hex_board for the player
        based on available transposition table values (only works with iterative deepening).
        Ordering of moves will help the alpha-beta methodology omit more nodes during search.

        If no iterative deepening is used the moves are simply uniformly scrambled so that
        all moves are uniformly likely to be the i'th expanded node in the minimax search.
        As the available moves are initially ordered by their generating function, this
        shuffling prevents biasedness in the search.

        :param hex_board: HexBoard Class for game-logic.
        :param moves: list All available moves at the current state of the HexBoard.
        :param player: int HexBoard player color.
        :return: list An ordered list of moves.
        :see: negamax_alpha_beta
        """
        # The moves are initially shuffled to prevent bias.
        np.random.shuffle(moves)

        # Move ordering can only be done if there are transposition tables from previous iterations.
        if not self.use_hashing or not len(self.hashtable) > 1:
            return moves  # No data to order moves on.

        move_scores = list()
        for move in moves:
            make_move(hex_board, move, player)      # --> To child Node
            hashed_value = self.zobrist(hex_board)  # Calculate hash of the child Node
            unmake_move(hex_board, move)            # --> Back to Parent Node

            # Get the most recent hashed score. (only applicable for iterative deepening)
            heuristic_values = [layer[hashed_value] for layer in self.transpositions if hashed_value in layer]
            move_scores.append(heuristic_values[-1] if len(heuristic_values) > 0 else 0)

        # Order Ascending by heuristic value such that the best moves are sorted at the front,
        # unexplored moves shuffled in the middle and detrimental moves at the very end.
        ordering = sorted(zip(move_scores, moves), key=lambda pair: pair[0], reverse=True)

        # Flattens the sorted list of (move_value, moves) list to
        # [(tuple of move_values), (tuple of moves)] and returns the now ordered (tuple of moves)
        return list(zip(*ordering))[1]

    def negamax_alpha_beta(self, hex_board, depth, alpha, beta, player):
        """
        Compute the heuristically evaluated Minimax optimal value-move-depth triple and
        return this triple to the previous (root) depth. Minimax is implemented
        using negamax along with alpha-beta pruning.

        Note that if at a root state multiple children are found with the highest score,
        the best node is then selected on value and the lowest evaluation depth. If both
        children have the same best heuristic value and evaluation depth, either one is
        equiprobable of being chosen.

        :param hex_board: HexBoard Class for game-logic.
        :param depth: int Current search depth.
        :param alpha: int Lower-bound on the heuristic value of a state.
        :param beta: int Upper-bound on the heuristic value of a state.
        :param player: int HexBoard player color.
        :return: (int, tuple, int) Minimax optimal heuristic value, Coordinate, and eval depth.
        :see: move_ordering
        :references: https://en.wikipedia.org/wiki/Minimax
        :references: https://en.wikipedia.org/wiki/Negamax
        :references: https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
        """
        hash_value = 0
        heuristic_best = self._undefined_node

        if self.use_hashing:
            # Fetch transposition table value if exists
            hash_value = self.zobrist(hex_board)

        if hash_value in self.hashtable:
            # Node already encountered and should NOT be expanded
            # Note that for iterative deepening only moves of the CURRENT
            # search depth can be found in the hashtable. Nodes encountered
            # at previous depths (= older age) are used for move ordering.
            return self.hashtable[hash_value], self.TERMINAL, depth

        elif self.evaluations > self.budget:
            # If no more function calls are available, return a default value.
            return self._undefined_node

        elif depth == 0 or hex_board.game_over or no_moves(hex_board):
            # Terminal Node or Depth Limit reached
            heuristic_best = (self.evaluate(hex_board, player), self.TERMINAL, depth)

        else:
            # Node should be expanded
            moves = available_moves(hex_board)
            moves = self.order_moves(hex_board, moves, player)

            for move in moves:
                # --> To child Node
                make_move(hex_board, move, player)

                # Expand into child Node
                recursive_value, _, eval_depth = self.negamax_alpha_beta(
                    hex_board, depth - 1, -beta, -alpha, hex_board.get_opposite_color(player))
                # Negate value due to players switching, format: (minmax-value, move)
                recursive_best = (-recursive_value, move, eval_depth)

                # --> Back to Parent Node
                unmake_move(hex_board, move)

                # Perform Minimax selection and alpha-beta pruning.
                # Select best child on value first, on evaluation depth second (shallow preferred)
                heuristic_best = max(heuristic_best, recursive_best,
                                     key=lambda x: (x[self._VALUE_IDX], x[self._EVAL_DEPTH_IDX]))
                alpha = max(alpha, heuristic_best[self._VALUE_IDX])
                if alpha >= beta:
                    break

        # Store calculated heuristic value into transposition table (if used)
        if self.use_hashing:
            self.hashtable[hash_value] = heuristic_best[self._VALUE_IDX]

        # Returns (value, (col, row), evaluation_depth)
        return heuristic_best

    def search(self, hex_board, depth, budget=None):
        """
        Perform fixed-depth minimax search. Puts an upper-bound on the budget
        if no budget is provided to prevent infinite computation time.
        :param hex_board: HexBoard Class for game-logic.
        :param depth: int The depth to search in the game-tree.
        :param budget: int Maximum amount of heuristic function evaluations
        :see: negamax_alpha_beta
        """
        # Calculate the maximum amount of nodes that can be searched at the given depth
        n_empty = len(available_moves(hex_board))
        max_nodes = np.math.factorial(n_empty) / np.math.factorial(np.max([0, n_empty - depth]))

        # Clip the budget to prevent infinite computation time
        self.budget = np.min([10 ** 9, max_nodes]) if not budget else np.min([budget, max_nodes])

        # Perform the search with fixed depth
        _, self.move, _ = self.negamax_alpha_beta(
            hex_board, depth, alpha=-np.inf, beta=np.inf, player=self.player)

        if self.move == self.TERMINAL:  # If search failed, choose a random move.
            self.move = np.random.choice(available_moves(hex_board))

    def iterative_deepening(self, hex_board, budget, depth=1, steps=1):
        """
        Search for a move using Minimax on the given HexBoard with increasing depth.
        This allows the algorithm to retain the best move from a previous depth such that
        the search can be safely terminated if stuck in higher depths.
        :param hex_board:HexBoard Class for game-logic.
        :param budget: int Maximum amount of heuristic function evaluations.
        :param depth: int Starting depth for the alpha-beta search.
        :param steps: int The increase to make for search depth at each iteration.
        :see: negamax_alpha_beta
        :references: https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search
        """
        # Made a class variable to keep track of actual depth during search -- e.g., for
        # transposition table age.
        self.itd_depth = depth

        # For irregular depths (depth > 1) pad the list of TTs with empty tables so that
        # the class can fetch the current iteration's TT using self.itd_depth.
        any(self.transpositions.append(dict()) for _ in range(depth-1))

        # In late game scenarios search-depth may exceed the max possible depth as
        # there are less positions to be searched. Bounding depth precludes redundant search.
        max_depth = len(available_moves(hex_board))

        best_node = self._undefined_node
        self.budget = budget
        while self.evaluations < budget and self.itd_depth <= max_depth:
            new_node = self.negamax_alpha_beta(
                hex_board, depth=self.itd_depth, alpha=-np.inf, beta=np.inf, player=self.player)

            # Don't keep the most recent move if the search failed or had to back-track prematurely.
            if new_node[self._MOVE_IDX] != self.TERMINAL and self.evaluations < budget:
                if new_node[self._VALUE_IDX] >= best_node[self._VALUE_IDX]:
                    best_node = new_node
                self.move = best_node[self._MOVE_IDX]

            # Debugging line:
            print("ID depth-evaluations:", self.itd_depth, self.evaluations,
                  "update:", new_node[self._MOVE_IDX] != self.TERMINAL and self.evaluations < budget)
            print("Found:",new_node[self._MOVE_IDX], new_node[self._VALUE_IDX])
            print("Keep:", self.move, best_node[self._VALUE_IDX])

            self.itd_depth += steps
            # Add a new emptpy transposition table for the next depths. The transposition tables
            # from previous layers are now solely used for move ordering.
            any(self.transpositions.append(dict()) for _ in range(steps))


class MCTSSearcher(Searcher):
    """
    Derived class of Searcher which implements the MCTS algorithm.
    This implementation additionally provides the option to memorize
    the subtree of earlier MCTS searches, so that a new search can
    exploit previously accumulated node statistics; which may improve
    empirical performance.
    """

    class Node(object):
        """
        Wrapper class for the Children of a boardstate that stores
        the statistics requisite for the MCTS algorithm.

        The class stores its children, its visit count, win count, etc.
        Additionally the class implements the methods required
        for Node selection, expansion, and updating.
        """

        def __init__(self, boardstate, player):
            """
            Initializes the wrapper class Node for the given boardstate.
            :param boardstate: HexBoard Class for game-logic.
            :param player: int HexBoard player color currently to move.
            """
            self.state = boardstate
            self.parentNode = self.move = None
            self.visits = self.wins = 0
            self.untriedMoves = available_moves(boardstate)
            self.childNodes = list()
            self.player = player

        def select(self, c_p, n_i):
            """
            Performs the child Selection step of the MCTS algorithm.
            Select a Node from the list self.childNodes based on its UCT-value
            :param c_p: float Exploration parameter for the UCT formula.
            :param n_i: int The amount of visits to the parent node of node.
            :return: Node Childnode of this class with the highest UCT value.
            :see: MCTSSearcher._uct
            """
            return max(self.childNodes, key=lambda node: MCTSSearcher._uct(node, c_p, n_i))

        def expand(self, move, state):
            """
            Performs the Child Expansion step of the MCTS algorihthm.
            Create a Node class from the given move and board-state. This function
            creates a Childnode for state (state is a child-state of 'self'). The child
            receives this class as its parent, so that the parent can be traced back later
            during backtracking. The child receives the move variable to memorize which move
            caused the state of the current class to reach the argument 'state'.
            Additonally the statistics of this class's children is updated.
            :param move: tuple Coordinates on the Hexboard = state that lead to this state.
            :param state: HexBoard Class for game-logic in the state after playing 'move'.
            :return: Node The newly created childnode of the current class.
            """
            child = MCTSSearcher.Node(state, state.get_opposite_color(self.player))
            child.move = move
            child.parentNode = self
            self.childNodes.append(child)
            self.untriedMoves.remove(move)
            return child

        def update(self, result):
            """
            Update the statistics of this class based on the result of a rollout/ random playout.
            The visit count is updated, and if the MCTSSearcher's player has won, the win count is
            also incremented.
            :param result: bool True if MCTSSearcher.player has won, otherwise False.
            """
            self.visits += 1
            if result:
                self.wins += 1

    def __init__(self, player, memorized_tree=None):
        """
        Initialize the MCTS searcher with a player perspective to score
        the explored nodes on. Additionally a subtree from a previous search
        can be provided to exploit the accumulated statistics from multiple searches.
        :param player: int HexBoard player color.
        :param memorized_tree: Node MCTS tree structure expanded from a previous search.
        """
        super().__init__()
        self.player = player
        self.memorized_tree = memorized_tree
        if self.memorized_tree:
            self.memorized_tree.parentNode = None  # Set the memorized tree to a Root node.

    @staticmethod
    def _uct(node, c_p, n_i):
        """
        Computes the upper confidence bound 1 applied to trees equation for a Node.
        :param node: Node Class for wrapping the children of the HexBoard state/ storing statistics for MCTS.
        :param c_p: float Exploration parameter for the UCT formula.
        :param n_i: int The amount of visits to the parent node of node.
        :return: UCT value
        :see: Node
        """
        return node.wins / node.visits + c_p * np.sqrt(np.log(n_i + 1) / node.visits)

    @staticmethod
    def _hex_playout_fill(hex_board, player):
        """
        Perform a rollout for the MCTS algorithm by completely filling the HexBoard
        uniformly at random with either players.

        All empty positions on the board are collected, scrambled, and uniformly
        divided to the player and its adversary. If the amount of moves modulo 2 = 1,
        the player will have an additional move. The board is filled with the
        divided moves and afterwards the winner is checked. This is possible
        due to Hex being a deterministic game. If one player wins, the other
        automatically loses. After determining the result, the board is emptied
        to its argument state 'hex_board', and the result is returned.

        This filling mimicks random-play, however is a magnitude more efficient than
        normal random play. Seeing as the board doesn't need to check who won
        after every move but only once after the game is finished. We found a
        10-fold speed-up over normal random-play using this method.

        :param hex_board: HexBoard Class for game-logic
        :param player: int HexBoard player color. Current player to move.
        :return: bool True if player wins after the random game.
        """
        # Generate all possible moves and permute them to
        # efficient select moves uniformly random.
        move_set = available_moves(hex_board)
        np.random.shuffle(move_set)

        # Evenly split the moves among the two players.
        # Array split always give the first element the residual
        # element if the length of the move list is uneven.
        player_moves, adversary_moves = np.array_split(move_set, 2)
        adversary = hex_board.get_opposite_color(player)

        # Fill the entire board with the permuted possible moves.
        any(emplace(hex_board, tuple(move), player) for move in player_moves)
        any(emplace(hex_board, tuple(move), adversary) for move in adversary_moves)

        # Get the result of the random playout. If False --> adversary won.
        player_won = hex_board.check_win(player)

        # Reset the board to its original state.
        any(unmake_move(hex_board, tuple(move)) for move in player_moves)
        any(unmake_move(hex_board, tuple(move)) for move in adversary_moves)

        return player_won  # And return the result...

    @staticmethod
    def _backtrack(state, node, result):
        """
        Perform the backtracking step of the MCTS algorithm.
        "Climb" from the child 'node' back to the rootnode while updating the visited
        nodes with the requisite statistics provided by 'result'. The board-state
        is simultaneously returned to the root-state.
        :param state: HexBoard Class for game-logic in the state of the expansion of 'node'.
        :param node: Node Class for wrapping the children of the HexBoard state/ storing statistics for MCTS.
        :param result: bool True if 'self.player' won, otherwise False.
        :return: Node Returns the node returned to its rootstate with updated statistics.
        """
        while node.parentNode:  # Only the rootstate has parentNode = None.
            node.update(result)
            unmake_move(state, node.move)
            node = node.parentNode

        return node

    @staticmethod
    def find_next_subtree(node, new_state):
        """
        Given a subtree Node and a HexBoard state that is exactly ONE move ahead of node,
        find the subtree of node belonging to state. This function is used for MCTS tree
        memorization without requiring external memorization of previously made moves.
        :param node: Node Class for wrapping the children of the HexBoard state/ storing statistics for MCTS.
       :param state: HexBoard Class for game-logic that is one game-state ahead of the tree in Node.
        :return:
        """
        node_moves = [move for move in node.untriedMoves] + [child.move for child in node.childNodes]
        moves = available_moves(new_state)

        # Get the difference in moves (i.e., the previous move).
        difference = set(node_moves) - set(moves)
        if not difference:
            return None

        # Find the MCTS subtree belonging to the previous move
        move_difference = difference.pop()
        subtree = [child for child in node.childNodes if child.move == move_difference]

        # If the child exists, subtree has only one element.
        return subtree[0] if subtree else None

    def search(self, hex_board, exploration=1.0, budget=1_000, monitor=False):
        """
        Performs the main procedure of the MCTS algorithm.
        :param hex_board: HexBoard Class for game-logic.
        :param exploration: float The exploration parameter for the expansion procedure (C_p).
        :param budget: int The amount of MCTS simulations to perform.
        :param monitor: bool Whether to print out intermediary progress and statistics.
        """
        # Initialize the rootstate with the subtree of a previous search or generate a new tree.
        node = self.memorized_tree if self.memorized_tree else MCTSSearcher.Node(hex_board, self.player)
        runtime = 0
        if monitor:
            runtime = time.time()

        # Budget != N. Budget is the amount of simulations to perform during this call of
        # the function. While N is the total amount of simulations. If the rootnode is
        # a subtree of a previous search, the total amount of simulations is the budget
        # plus the amount of visits to the current rootnode.
        for sim_i in range(node.visits + 1, node.visits + budget + 1):
            if monitor:
                if sim_i % 1_000 == 0:
                    print("At iteration {} with ~{:.4f} seconds per iteration".format(
                        sim_i, (time.time() - runtime) / sim_i))

            to_move = self.player

            # Selection
            while not node.untriedMoves and node.childNodes:
                node = node.select(exploration, node.visits)
                make_move(hex_board, node.move, to_move)
                to_move = hex_board.get_opposite_color(to_move)

            # Expand
            if node.untriedMoves:
                move = node.untriedMoves[np.random.randint(len(node.untriedMoves))]
                make_move(hex_board, move, to_move)
                to_move = hex_board.get_opposite_color(to_move)
                node = node.expand(move, hex_board)

            # Playout
            # If to_move is the adversary than the result should be loss.
            # If to_move is the player than the result should be win.
            result = MCTSSearcher._hex_playout_fill(hex_board, to_move)

            # Backpropagate
            # Uses result = True if self.player won.
            node = MCTSSearcher._backtrack(hex_board, node, (to_move == self.player) == result)

            # Update the currently best move after every iteration and the tree.
            self.memorized_tree = max(node.childNodes, key=lambda n: n.visits)
            self.move = self.memorized_tree.move

        if monitor:
            runtime = time.time() - runtime
            print("{} iterations done in {:.4f} seconds with {:.4f} seconds per iteration".format(
                budget, runtime, runtime / budget))
            print("Best move: {} with a visit count of {} and a winrate of {:.3f}".format(
                self.move, self.memorized_tree.visits, self.memorized_tree.wins / self.memorized_tree.visits))
