"""
This Python file implements the Policies for game-playing agents that
is used for playing the game of Hex. All Policies are derived from the
Policy class which serves as the format that a policy should adhere to:
which means, provided a state s calculate pi(s) to obtain an action a.

The currently implemented Policies are:
 - Manual policy, choose an action by user input.
 - Random policy, choose an action uniformly random from available moves.
 - Deterministic policy, choose the first action in the available moves.
 - Minimax policy, choose actions by the Minimax Algorithm.
 - MCTS Policy, choose actions by the Monte Carlo Tree Search algorithm.
 - AlphaZero Policy, using the AlphaZero General backend.

:version: FINAL
:date: 15-05-2020
:author: Joery de Vries
:edited by: Joery de Vries, Oliver Konig, Siyuan Dong
"""

from ast import literal_eval

import numpy as np

from alphazero.hex.HexGame import HexGame
from alphazero.MCTS import MCTS
from alphazero.utils import dotdict

from .hex_search import MinimaxSearcher, MCTSSearcher
from .hex_utils import available_moves


def run(function, args):
    function(args)


class Policy(object):
    """
    Abstract class to provide a format for a move-selecting policy.

    Given a state s (=HexBoard) a derived class should implement a
    policy algorithm/ method pi to generate the action a to return.
    At the generation of a move the variable calls should be incremented.
    """

    def __init__(self):
        self.calls = 0
        self.perspective = None

    def set_perspective(self, perspective):
        """
        (Re-)Set the player perspective of the policy. In other words,
        configure the policy to choose moves for the given player.
        This variable is important for search engines such as minimax
        which would, if not correctly set, find optimal moves for the
        adversary.
        :param perspective: int Player perspective.
        """
        self.perspective = perspective

    def generate_move(self, hex_board):
        """
        Given a state hex_board, use the policy to return a move.
        This function always increments the self.calls variable.
        :param hex_board: HexBoard Class for game-logic.
        :return: tuple Coordinate on the HexBoard to move to.
        """
        self.calls += 1
        return None


class ManualPolicy(Policy):
    """
    Derived class of Policy to choose actions based on (manual)
    user input.
    """

    def __init__(self):
        super().__init__()

    def generate_move(self, hex_board):
        """
        Given a hex game-board, ask the user for a valid action/ move.
        :param hex_board: HexBoard Class for game-logic.
        :return: tuple Coordinate on the HexBoard to move to.
        """
        self.calls += 1
        moves = available_moves(hex_board)

        in_move = None
        while in_move not in moves:
            try:
                in_move = literal_eval(input("Input a valid move (row, col): "))
            except Exception:
                pass

        return in_move


class RandomPolicy(Policy):
    """
    Derived class of Policy to choose actions uniformly random.
    """

    def __init__(self, seed=None):
        """
        For experimentation purposes a random seed can be provided.
        :param seed: int Pseudo-RNG seed
        """
        super().__init__()
        self.seed = seed

    def generate_move(self, hex_board):
        """
        Given a hex game-board, calculate all available moves and
        choose one action/ move uniformly random.
        :param hex_board: HexBoard Class for game-logic.
        :return: tuple Coordinate on the HexBoard to move to.
        """
        self.calls += 1
        if self.seed is not None:
            np.random.seed(self.seed)
        moves = available_moves(hex_board)
        return moves[np.random.randint(len(moves))] if len(moves) > 0 else None


class DeterministicPolicy(Policy):
    """
    Derived class of Policy to choose actions naively deterministic.
    This policy always chooses the same (valid) move on a given state.
    """

    def __init__(self):
        super().__init__()

    def generate_move(self, hex_board):
        """
        Given a hex game-board, calculate all available moves and
        choose the action/ move at the first position in the list.
        :param hex_board: HexBoard Class for game-logic.
        :return: tuple Coordinate on the HexBoard to move to.
        """
        self.calls += 1
        moves = available_moves(hex_board)
        return moves[0] if len(moves) > 0 else None


class MinimaxPolicy(Policy):
    """
    Derived class of Policy to choose actions based on the Minimax Algorithm.
    """

    def __init__(self, heuristic, depth=0, itd=False, transpositions=False, budget=1e9, perspective=None, timespan=None):
        """
        The Minimax algorithm must be configured with the following:
         - A heuristic scoring function that can score a given board-state.
         - A fixed depth larger than zero or iterative deepening.
         - A player color/ perspective to distinguish between adversaries in the Min-Max nodes.
         - Whether to use hashing/ transposition tables.
         - A maximum budget for calling the heuristic evaluation function.
        :param heuristic: HexHeuristic Implementation of a heuristic evaluation function.
        :param depth: int Depth to search if fixed-depth search is used.
        :param itd: boolean True to use iterative depth-search.
        :param transpositions: boolean True to use Zobrist hashing/ hash-tables.
        :param budget: int Maximum amount of evaluations of the heuristic function.
        :param timespan: int Maximum amount of seconds the search can run. (not yet implemented)
        :see: HexHeuristic from .hex_heuristics
        """
        super().__init__()
        self.perspective = perspective
        self.depth = depth
        self.heuristic = heuristic
        self.transpositions = transpositions
        self.itd = itd
        self.budget = budget
        self.timespan = timespan

    def generate_move(self, hex_board):
        """
        With the given configuration for the minimax search and
        a currently provided game-state (=hex_board) call the
        Minimax-Searcher class to perform Minimax Search to find
        a move.
        :param hex_board: HexBoard Class for game-logic.
        :return: tuple Coordinate on the HexBoard to move to.
        :see: MinimaxSearcher from .hex_search
        """
        self.calls += 1
        if not self.depth > 0 and not self.itd:
            raise Exception("No search specifications given.")

        if self.perspective is None:
            raise Exception("No search/ player perspective given.")

        searcher = MinimaxSearcher(self.perspective, self.heuristic)

        if self.transpositions:
            searcher.initalize_transposition(hex_board.size)

        # TODO Create process thread to run search concurrently. Terminate search after timespan.
        if self.itd:
            if self.depth:  # TODO Implement a step parameter in the policy for IDTT.
                searcher.iterative_deepening(hex_board, self.budget, self.depth)
            else:
                searcher.iterative_deepening(hex_board, self.budget)
        else:
            searcher.search(hex_board, self.depth, self.budget)

        return searcher.get_move()


class MCTSPolicy(Policy):
    """
    Derived class of Policy to choose actions based on the MCTS Algorithm.
    """

    # Variable to store the best MCTS-subtree of a search.
    memorized_tree = None

    def __init__(self, exploration, budget, memorize=False, monitor=False):
        """
        The MCTS algorihtm must be configured with the parameters for the
        Upper-Confidence-bound 1 applied to Trees (UCT) formula for child
        selection and expansion.
        :param exploration: float The exploration parameter of UCT (C_p).
        :param budget: int The amount of MCTS simulations to perform (N).
        :param memorize: bool Whether to exploit the statistics from previous searches.
        :param monitor: bool Whether to print out intermediary statistics during search.
        """
        super().__init__()
        self.exploration = exploration
        self.budget = budget
        self.memorize = memorize
        self.monitor = monitor

    def generate_move(self, hex_board):
        """
        With the given parameters for the MCTS procedure and a currently
        provided game-state (=hex_board) call the MCTSSearcher class to
        perform Monte Carlo Tree Search to choose a move for the current player.
        :param hex_board: HexBoard Class for game-logic.
        :return: tuple Coordinate on the HexBoard to move to.
        :see: MCTSSearcher from .hex_search
        """
        self.calls += 1
        if self.perspective is None:
            raise Exception("No search/ player perspective given.")

        if self.memorize and self.memorized_tree:
            self.memorized_tree = MCTSSearcher.find_next_subtree(self.memorized_tree, hex_board)

        searcher = MCTSSearcher(self.perspective, self.memorized_tree)

        # TODO Create process thread to run search concurrently. Terminate search after timespan.
        searcher.search(hex_board, self.exploration, self.budget, self.monitor)

        if self.memorize:
            self.memorized_tree = searcher.memorized_tree

        return searcher.get_move()


class AlphaZeroPolicy(Policy):
    """
    Derived class of Policy to choose actions based on the AlphaZero Algorithm.
    """

    def __init__(self, exploration, budget, model, boardsize, temperature=0):
        """
        The AlphaZero algorihtm must be configured with the parameters for the
        PUCT formula for childnode selection and expansion along with a search budget.
        Also the neural network must be provided along with the boardsize in order
        to initialize the MCTS backend of AlphaZero.
        :param exploration: float The exploration parameter of PUCT (c_puct).
        :param budget: int The amount of MCTS simulations to perform (N).
        :param model: keras.Model A keras backend model that guides MCTS's search
        :param boardsize: int Size of the game board in order to initialize MCTS.
        :param temperature: int Governs the degree of exploration. (0 = greedy)
        """
        super().__init__()
        self.args = dotdict({'numMCTSSims': budget, 'cpuct': exploration})
        self.model = model
        self.game = HexGame(boardsize)
        self.searcher = MCTS(self.game, self.model, self.args)
        self.temperature = temperature  # temp=0 implies greedy actions

    def generate_move(self, hex_board):
        """
        With the given parameters for the AlphaZero procedure and a currently
        provided game-state (=hex_board) call the backend MCTS class to
        perform forward search guided by the neural network to choose
        a move for the current player.
        :param hex_board: HexBoard Class for game-logic.
        :return: tuple Coordinate on the HexBoard to move to.
        :see: MCTSSearcher from .hex_search
        """
        self.calls += 1
        if self.perspective is None:
            raise Exception("No search/ player perspective given.")

        # The neural network model only learns in an uniform player perspective.
        # Hence we first alter the symmetry of the board according to the current player.
        # The returned move is of course transformed to the perspective of the
        # board that was provided in the function's argument.
        search_board = self.game.getCanonicalForm(np.copy(hex_board.board), self.perspective)
        pi = self.searcher.getActionProb(search_board, temp=self.temperature)
        move_idx = np.argmax(pi)

        move = (move_idx // hex_board.size, move_idx % hex_board.size)
        if self.perspective == -1:  # Canonical form finds a move on a transposed board.
            move = move[::-1]

        return move
