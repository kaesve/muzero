"""
Contains logic for performing Monte Carlo Tree Search having access to the environment.

The class is an adaptation of AlphaZero-General's MCTS search to accommodate non-adversarial environments (MDPs).
We utilize the MinMax scaling of backed-up rewards for the UCB formula and (by default) compute the UCB using
the formula proposed by the MuZero algorithm. The MCTS returns both the estimated root-value and action probabilities.
The MCTS also discounts backed up rewards given that gamma < 1.

Notes:
 -  Adapted from https://github.com/suragnair/alpha-zero-general
 -  Base implementation done.
 -  Documentation 15/11/2020
"""
import typing

import numpy as np

from AlphaZero.AlphaNeuralNet import AlphaZeroNeuralNet
from utils import DotDict
from utils.selfplay_utils import MinMaxStats, GameHistory, GameState

EPS = 1e-8


class MCTS:
    """
    This class handles the MCTS tree while having access to the environment logic.
    """
    CANONICAL: bool = False  # Whether to compute the UCB formula using AlphaZero's formula (true) or MuZero's formula.

    def __init__(self, game, neural_net: AlphaZeroNeuralNet, args: DotDict) -> None:
        """
        Initialize all requisite variables for performing MCTS for AlphaZero.
        :param game: Game Implementation of Game class for environment logic.
        :param neural_net: AlphaNeuralNet Implementation of AlphaNeuralNet class for inference.
        :param args: DotDict Data structure containing parameters for the tree search.
        """
        self.game = game
        self.neural_net = neural_net
        self.args = args

        # Static helper variables.
        self.single_player = game.n_players == 1
        self.action_size = game.getActionSize()

        # Gets reinitialized at every search
        self.minmax = MinMaxStats(self.args.minimum_reward, self.args.maximum_reward)

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Ssa = {}  # stores state transitions for s, a
        self.Rsa = {}  # stores R values for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)
        self.Es = {}   # stores game.getGameEnded ended for board s
        self.Vs = {}   # stores game.getValidMoves for board s

    def clear_tree(self) -> None:
        """ Clear all statistics stored in the current search tree """
        self.Qsa, self.Ssa, self.Rsa, self.Nsa, self.Ns, self.Ps, self.Es, self.Vs = [{} for _ in range(8)]

    def initialize_root(self, state: GameState, trajectory: GameHistory) -> typing.Tuple[bytes, float]:
        """
        Perform initial inference for the root state and perturb the network prior with Dirichlet noise.
        Additionally mask the illegal moves in the network prior and initialize all statistics for starting the
        MCTS search.
        :param state: GameState Data structure containing the current state of the environment.
        :param trajectory: GameHistory Data structure containing the entire episode trajectory of the agent(s).
        :return: tuple (hash, root_value) The hash of the environment state and inferred root-value.
        """
        network_input = trajectory.stackObservations(self.neural_net.net_args.observation_length, state.observation)
        pi_0, v_0 = self.neural_net.predict(network_input)

        s_0 = self.game.getHash(state)

        # Add Dirichlet Exploration noise
        noise = np.random.dirichlet([self.args.dirichlet_alpha] * len(pi_0))
        self.Ps[s_0] = noise * self.args.exploration_fraction + (1 - self.args.exploration_fraction) * pi_0

        # Mask the prior for illegal moves, and re-normalize accordingly.
        self.Vs[s_0] = self.game.getLegalMoves(state)

        self.Ps[s_0] *= self.Vs[s_0]
        self.Ps[s_0] = self.Ps[s_0] / np.sum(self.Ps[s_0])

        # Sum of visit counts of the edges/ children and legal moves.
        self.Ns[s_0] = 0

        return s_0, v_0

    def compute_ucb(self, s: bytes, a: int, exploration_factor: float) -> float:
        """
        Compute the UCB for an edge (s, a) within the MCTS tree:

            PUCT(s, a) = MinMaxNormalize(Q(s, a)) + P(s, a) * sqrt(visits_parent / (1 + visits_s)) * exploration_factor

        Where the exploration factor is either the exploration term of MuZero (default) or a float c_1.

        Illegal edges are returned as zeros. The Q values within the tree are MinMax normalized over the
        accumulated statistics over the current tree search.

        :param s: hash Key of the current state inside the MCTS tree.
        :param a: int Action key representing the path to reach the child node from path (s, a)
        :param exploration_factor: float Pre-computed exploration factor from the MuZero PUCT formula.
        :return: float Upper confidence bound with neural network prior
        """
        if s in self.Vs and not self.Vs[s][a]:
            return 0

        visit_count = self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
        q_value = self.minmax.normalize(self.Qsa[(s, a)]) if (s, a) in self.Qsa else 0
        c_children = np.max([self.Ns[s], 1e-8])  # Ensure that prior doesn't collapse to 0 if s is new.

        # Exploration
        if self.CANONICAL:
            # Standard PUCT formula from the AlphaZero paper
            ucb = self.Ps[s][a] * np.sqrt(c_children) / (1 + visit_count) * self.args.c1
        else:
            # The PUCT formula from the MuZero paper
            ucb = self.Ps[s][a] * np.sqrt(c_children) / (1 + visit_count) * exploration_factor

        ucb += q_value  # Exploitation
        return ucb

    def runMCTS(self, state: GameState, trajectory: GameHistory, temp: int = 1) -> typing.Tuple[np.ndarray, float]:
        """
        This function performs 'num_MCTS_sims' simulations of MCTS starting from the provided root GameState.

        Before the search we only clear statistics stored inside the MinMax tree. In this way we ensure that
        reward bounds get refreshed over time/ don't get affected by strong reward scaling in past searches.
        This implementation, thus, reuses state state transitions from past searches. This may influence memory usage.

        Our estimation of the root-value of the MCTS tree search is based on a sample average of each backed-up
        MCTS value. This means that this estimate represents an on-policy estimate V^pi.

        Illegal moves are masked before computing the action probabilities.

        :param state: GameState Data structure containing the current state of the environment.
        :param trajectory: GameHistory Data structure containing the entire episode trajectory of the agent(s).
        :param temp: float Visit count exponentiation factor. A value of 0 = Greedy, +infinity = uniformly random.
        :return: tuple (pi, v) The move probabilities of MCTS and the estimated root-value of the policy.
        """
        # Refresh value bounds in the tree
        self.minmax.refresh()

        # Initialize the root variables needed for MCTS.
        s_0, v_0 = self.initialize_root(state, trajectory)

        # Aggregate root state value over MCTS back-propagated values
        v_search = sum([self._search(state, trajectory) for _ in range(self.args.num_MCTS_sims) - 1])
        v = (v_0 + (v_search if self.single_player else -v_search)) / self.args.num_MCTS_sims

        # MCTS Visit count array for each edge 'a' from root node 's_0'.
        counts = np.array([self.Nsa[(s_0, a)] if (s_0, a) in self.Nsa else 0 for a in range(self.action_size)])

        if temp == 0:  # Greedy selection. One hot encode the most visited paths (randomly break ties).
            move_probabilities = np.zeros(len(counts))
            move_probabilities[np.argmax(counts + np.random.randn(len(counts)) * 1e-8)] = 1
        else:
            counts = np.power(counts, 1. / temp)
            move_probabilities = counts / np.sum(counts)

        return move_probabilities, v

    def _search(self, state: GameState, trajectory: GameHistory, path: typing.Tuple[int, ...] = tuple()) -> float:
        """
        Recursively perform MCTS search inside the actual environments with search-paths guided by the PUCT formula.

        Selection chooses an action for expanding/ traversing the edge (s, a) within the tree search.
        The exploration_factor for the PUCT formula is computed within this function for efficiency:

            exploration_factor = c1 * log(visits_s + c2 + 1) - log(c2)

        Setting AlphaMCTS.CANONICAL to true sets exploration_factor just to c1.

        If an edge is expanded, we perform a step within the environment (with action a) and observe the state
        transition, reward, and infer the new move probabilities, and state value. If an edge is traversed, we simply
        look up earlier inferred/ observed values from the class dictionaries.

        During backup we update the current value estimates of an edge Q(s, a) using an average, we additionally
        update the MinMax statistics to get reward/ value boundaries for the PUCT formula. Note that backed-up
        values get discounted for gamma < 1. For adversarial games, we negate the backed up value G_k at each backup.

        The actual search-path 'path' is kept as a debugging-variable, it currently has no practical use. This method
        may raise a recursion error if the environment creates cycles, this should be highly improbable for most
        environments. If this does occur, the environment can be altered to terminate after n visits to some cycle.

        :param state: GameState Numerical prediction of the state by the encoder/ dynamics model.
        :param trajectory: GameHistory Data structure containing all observations until the current search-depth.
        :param path: tuple of integers representing the tree search-path of the current function call.
        :return: float The backed-up discounted/ Monte-Carlo returns (dependent on gamma) of the tree search.
        :raises RecursionError: When cycles occur within the search path, the search can get stuck *ad infinitum*.
        """
        s = self.game.getHash(state)

        ### SELECTION
        # pick the action with the highest upper confidence bound
        exploration_factor = self.args.c1 + np.log(self.Ns[s] + self.args.c2 + 1) - np.log(self.args.c2)
        confidence_bounds = [self.compute_ucb(s, a, exploration_factor) for a in range(self.action_size)]
        a = np.argmax(self.Vs[s] * np.asarray(confidence_bounds)).item()  # Get argmax as scalar

        if (s, a) not in self.Ssa:  ### ROLLOUT
            next_state, reward = self.game.getNextState(state, a, clone=True)
            s_next = self.game.getHash(next_state)

            if s_next not in self.Es:
                self.Es[s_next] = self.game.getGameEnded(next_state)

            if self.Es[s_next]:  # Leaf node
                value = 0 if self.single_player else -self.Es[s_next]
                self.Rsa[(s, a)] = 0
            else:
                # Build network input for inference
                network_input = trajectory.stackObservations(self.neural_net.net_args.observation_length,
                                                             state.observation)
                prior, value = self.neural_net.predict(network_input)

                # Current depth statistics
                self.Rsa[(s, a)], self.Ssa[(s, a)] = reward, next_state
                # Next depth statistics
                self.Ps[s_next], self.Ns[s_next], self.Vs[s_next] = prior, 0, self.game.getLegalMoves(next_state)
                value = value if self.single_player else -value  # Alternate value perspective for adversary.

        else:  ### EXPANSION
            trajectory.observations.append(state.observation)   # Build up an observation trajectory inside the tree
            value = self._search(self.Ssa[(s, a)], trajectory, path + (a,))
            trajectory.observations.pop()                       # Clear tree observation trajectory when backing up

        ### BACKUP
        gk = self.Rsa[(s, a)] + self.args.gamma * value  # (Discounted) Value of the current node

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + gk) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = gk
            self.Nsa[(s, a)] = 1

        self.minmax.update(self.Qsa[(s, a)])
        self.Ns[s] += 1

        return gk if self.single_player else -gk
