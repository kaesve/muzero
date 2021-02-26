"""
This Python file defines the Classes for Heuristic scoring of provided
hex-board game states. All implemented heuristics are derived from HexHeuristic
which serves as the format that a heuristic scoring function should adhere to.

The currently implemented heuristic scoring functions of the hex-game are:
 - RandomHeuristic, evaluate a state by returning a random integer.
 - DeterministicHeuristic, evaluate a state by always returning the same integer.
 - DijkstraHeuristic, evaluate a state by means of computing the least-cost path
   to a terminal (winning) game state.

:version: FINAL
:date: 07-02-2020
:author: Joery de Vries and Ken Voskuil
:edited by: Joery de Vries and Ken Voskuil
"""

import numpy as np

from .hex_utils import placed_positions


class HexHeuristic(object):
    """
    Abstract class to provide a format for a state evaluating heuristic.

    Given a state s, compute a score h(s) that approximates and quantifies the
    beneficiality of that particular state.
    """

    def __init__(self):
        pass

    def evaluate(self, hex_board, player):
        """
        Given a state compute the heuristic value of the state for the
        perspective of the provided HexBoard player.
        :param hex_board: HexBoard Class for game-logic.
        :param player: int HexBoard player color.
        :return: int Quantified beneficiality of the given state for the player.
        """
        return None


class RandomHeuristic(HexHeuristic):
    """
    Derived class of HexHeuristic to quantify a state by generating random integers.
    """

    def __init__(self, low, high, seed=None):
        """
        Initialize the Random heuristic with a low and high boundary for
        the random integer generation.
        For experimentation purposes a random seed can be provided.
        :param low: int Lower boundary for the random heuristic value
        :param high: int Upper boundary for the random heuristic value
        :param seed: int Pseudo-RNG seed
        """
        super().__init__()
        self.low = low
        self.high = high
        self.seed = seed

    def evaluate(self, hex_board, player):
        """
        Given a state simply compute a random integer as the heuristic
        value between provided boundaries.
        :param hex_board: HexBoard Class for game-logic.
        :param player: int HexBoard player color.
        :return: int Random integer between low and high.
        """
        if self.seed:
            np.random.seed(self.seed)
        return np.random.randint(self.low, self.high)


class ConstantHeuristic(HexHeuristic):
    """
    Derived class of HexHeuristic to quantify a state by always returning
    the same constant value.
    """

    def __init__(self, value):
        """
        Initialize the constant heuristic with the value to always return.
        :param value: Number Value to always return at evaluation.
        """
        super().__init__()
        self.value = value

    def evaluate(self, hex_board, player):
        """
        Given a state simply compute a random integer as the heuristic
        value between provided boundaries.
        :param hex_board: HexBoard Class for game-logic.
        :param player: int HexBoard player color.
        :return: int Constant heuristic value.
        """
        return self.value


class DijkstraHeuristic(HexHeuristic):
    """

    """

    class HexboardGraph:
        """
        Provides a graph representation for the HexBoard for Graph-Search.

        The HexBoardGraph is represented as:
                Top
                |
        Left - GRID - Right
                |
            Bottom

        Where Left and Right are the Root and Goal states for red, respectively.
        Where Top and Bottom are the Root and Goal states for blue, respectively.

        This class provides methods to expand the Root, Goal, and default positions.
        For example, Left expands into the n coordinates in the first column of the
        grid and Bottom expands into the n coordinates in the last row of the grid.

        Nodes that are expanded are either empty or belong to the selected player.

        The class also scores nodes by the costs to reach them, which is either 1 if
        the node is empty or 0 if the node is a Root, Goal, or an already conquered node.

        :see: HexBoard in .hex_skeleton
        """

        def __init__(self, hex_board):
            """
            Initialize a Graph representation of the provided HexBoard.
            :param hex_board: HexBoard Class for game-logic.
            """
            self.hex_board = hex_board
            self.left = (0, -1)
            self.top = (-1, 0)
            self.right = (0, hex_board.size)
            self.bottom = (hex_board.size, 0)

        @property
        def red_start(self):
            return self.left

        @property
        def red_end(self):
            return self.right

        @property
        def blue_start(self):
            return self.top

        @property
        def blue_end(self):
            return self.bottom

        def expand(self, node, color):
            """
            Generate all neighbouring positions of the HexBoardGraph at the
            given node (pos) that can be reached by the player (color)
            :param node: tuple Coordinate on the HexBoardGraph.
            :param color: int Player color.
            :return: set All moves/ nodes that can be reached by color at node.
            """
            if node == self.left:      # Gets all positions on the left side of the board
                neighbors = [(i, 0) for i in range(self.hex_board.size)]
            elif node == self.right:   # Gets all positions on the right side of the board
                neighbors = [(i, self.hex_board.size - 1) for i in range(self.hex_board.size)]
            elif node == self.top:     # Gets all positions on the top of the board
                neighbors = [(0, i) for i in range(self.hex_board.size)]
            elif node == self.bottom:  # Gets all positions on the bottom of the board
                neighbors = [(self.hex_board.size - 1, i) for i in range(self.hex_board.size)]
            else:
                neighbors = self.hex_board.get_neighbors(node)

                # Add a player's Source/ Goal nodes to neighbors if node is adjacent.
                if node[0] == 0 and color == self.hex_board.BLUE:
                    neighbors.append(self.top)
                elif node[0] == self.hex_board.size - 1 and color == self.hex_board.BLUE:
                    neighbors.append(self.bottom)
                if node[1] == 0 and color == self.hex_board.RED:
                    neighbors.append(self.left)
                elif node[1] == self.hex_board.size - 1 and color == self.hex_board.RED:
                    neighbors.append(self.right)

            # Filter out board coordinates on move-validity.
            valid_nodes = set()
            for node in neighbors:
                # Terminal/ Source nodes are always valid as neighbors
                if node == self.left or node == self.right or node == self.bottom or node == self.top:
                    valid_nodes.add(node)

                # Empty or friendly nodes are valid to move to
                elif self.hex_board.is_empty(node) or self.hex_board.is_color(node, color):
                    valid_nodes.add(node)

            return valid_nodes

        def costs(self, node, color):
            """
            Evaluates the costs of moving to the provided node for the given
            player (color). A player can not move to nodes from the adversary.
            :param node: tuple Coordinate on the HexBoardGraph.
            :param color: int Player color.
            :return: int 0 If the node is from the ally or a Root/ Goal state, 1 otherwise.
            """
            if node == self.red_start or node == self.red_end or \
                    node == self.blue_start or node == self.blue_end:
                return 0  # Goal or Source nodes have no costs.
            elif self.hex_board.is_color(node, color):
                return 0  # If a piece is already placed --> no costs.
            else:
                return 1  # If Empty = piece must be placed --> 1 costs.

    def __init__(self):
        super().__init__()

    def evaluate(self, hex_board, player, discount=True):
        """
        Evaluate a given board state by computing the remaining distance
        that the ally and the adversary need to take and by subtracting the
        ally's advantage from the adversary's advantage.
        In other words, if player = RED, and RED needs to place (optimally)
        3 more pieces to reach a terminal winning game-state, and BLUE needs 5,
        then the heuristic score is 5 - 3 = 2.

        The mentioned distances are computed using dijkstra/ uniform cost search.
        For this reason the hex_board is represented as a graph using the
        HexBoardGraph wrapper.

        If the winning terminal game-state can't be reached, the costs will be clipped
        to the maximum cost possible (= board_size ^ 2).

        This heuristic scoring function quantifies how much 'closer' a player is from
        winning the game. Negative values, thus, imply that the adversary is 'closer'.
        :param hex_board: HexBoard Class for game-logic.
        :param player: int HexBoard player color.
        :param discount: bool Whether to discount one cost for the player that is to move.
        :return: int Remaining costs of the adversary - Remaining costs of the ally.
        :see: dijkstra_search, HexBoardGraph
        :references: http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
        """
        graph = self.HexboardGraph(hex_board)

        # Calculate heuristic costs (advantage) for each player.
        # That is, how many moves does the given player need to make
        # to reach a winning end-game state not considering an adversary.
        red_remaining_costs = self.dijkstra_search(
            graph, graph.red_start, graph.red_end, hex_board.RED)
        blue_remaining_costs = self.dijkstra_search(
            graph, graph.blue_start, graph.blue_end, hex_board.BLUE)

        # Clip infinite costs to maximum possible length
        red_remaining_costs = min(red_remaining_costs, hex_board.size ** 2)
        blue_remaining_costs = min(blue_remaining_costs, hex_board.size ** 2)

        # If the heuristic should discount, that is, if red is to move next we can
        # decrement one remaining move for red. This is helpful for near-terminal states.
        if discount:
            n_red = len(placed_positions(hex_board, hex_board.RED))
            n_blue = len(placed_positions(hex_board, hex_board.BLUE))
            if n_red > n_blue:  # Blue is to move
                blue_remaining_costs -= 1
            else:  # n_blue == n_red ==> Red is to move
                red_remaining_costs -= 1

        # Debugging lines.
        # print("red costs:", red_remaining_costs)
        # print("blue costs:", blue_remaining_costs)

        # The heuristic score is returned such that the remaining costs that
        # the adversary has to take should be LARGE and the remaining costs that
        # the ally must make must be SMALL. I.e., Adversary_costs - Ally_costs.
        if player == hex_board.RED:
            if red_remaining_costs <= 0:
                return hex_board.size ** 2  # Win
            elif blue_remaining_costs <= 0:
                return -hex_board.size ** 2  # Loss
            else:
                return blue_remaining_costs - red_remaining_costs
        else:
            if blue_remaining_costs <= 0:
                return hex_board.size ** 2  # Win
            elif red_remaining_costs <= 0:
                return -hex_board.size ** 2  # Loss
            else:
                return red_remaining_costs - blue_remaining_costs

    @staticmethod
    def dijkstra_search(graph, start, goal, color):
        """
        Implements a simple form of Dijkstra's algorithm
        Note that no ordering on the nodes (priority queue) is implemented in the search.
        :param graph: HexBoardGraph Graph representation of the HexBoard.
        :param start: tuple Coordinate of the source node on the HexBoardGraph.
        :param goal: tuple Coordinate of the goal node on the HexBoardGraph.
        :param color: int HexBoard player color for generating node Neighbours.
        :return: int The minimum amount of steps to reach goal from start.
        :references: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
        :see: DijkstraHeuristic
        """
        distance = {start: 0}        # Track least-cost distances between nodes.
        predecessor = {start: None}  # Track node-predecessors for the path.

        # The set of yet unexplored nodes. (Made a set to prevent loops)
        frontier = {start}

        while len(frontier) > 0:  # While frontier not empty
            node = frontier.pop()  # Pops an arbitrary element from the set

            for neighbor in graph.expand(node, color):
                # The cost is the currently accumulated cost of the node
                # plus the costs of moving to the current neighbor.
                cost = distance[node] + graph.costs(node, color)
                if neighbor not in distance:
                    distance[neighbor] = np.inf

                # If the distance to this neighbour is lower than a currently
                # known distance (or infinity) --> update the frontier.
                if cost < distance[neighbor]:
                    distance[neighbor] = cost
                    predecessor[neighbor] = node
                    frontier.add(neighbor)

        # If the goal is reachable return the least-cost distance, infinity otherwise.
        return distance[goal] if goal in distance else np.inf
