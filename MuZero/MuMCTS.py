"""
Contains logic for performing Monte Carlo Tree Search within the neural network embedding.
The class essentially recursively unrolls the recurrent MuZero neural network based on actions
sampled from the upper confidence bound formula. The MCTS returns the estimated root-value and action probabilities.

Notes:
 -  This implementation of MCTS performs the tree search in a recursive fashion; opposed to the common object
    oriented method. State transitions, values, rewards, etc. are thus stored in a dictionary that is dependent
    on a hash. We use the predicted latent-state by the encoder/ dynamics model for this hash which means that
    the model must be conditioned to assert non-trivial state transitions. The conditioning must prevent that
    s_k+1 = s_k = dynamics(s_k, a_k), i.e., a cycle in the MCTS tree; cycles can result in unexpectedly long
    search runtimes or recursion errors. Preventing this can take the form of changing the neural architecture,
    adding noise, among other methods. See existing implementations for examples.

    Hashes for the tree search are computed using the (un-rounded) numpy bytes data-buffer along with the action
    search-path inside the tree as a tuple: (bytes of numpy array data, tuple of action integers)

 -  Base implementation done.
 -  Documentation 15/11/2020
"""
import typing

import numpy as np

from MuZero.MuNeuralNet import MuZeroNeuralNet
from utils import DotDict
from utils.selfplay_utils import MinMaxStats, GameHistory, GameState

EPS = 1e-8


class MuZeroMCTS:
    """
    This class handles the MCTS tree within the neural network embedding.
    """

    def __init__(self, game, neural_net: MuZeroNeuralNet, args: DotDict) -> None:
        """
        Initialize all requisite variables for performing MCTS for MuZero.
        :param game: Game Implementation of Game class for environment logic.
        :param neural_net: MuNeuralNet Implementation of MuNeuralNet class for inference.
        :param args: DotDict Data structure containing parameters for the tree search.
        """
        self.game = game
        self.neural_net = neural_net
        self.args = args

        # Static helper variables to remove references to the 'game' object.
        self.single_player = game.n_players == 1
        self.action_size = game.getActionSize()

        # Gets reinitialized at every search
        self.minmax = MinMaxStats(self.args.minimum_reward, self.args.maximum_reward)

        # Note. In the tables below, 's' is defined as a tuple (s_tensor, tree_path).
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Ssa = {}  # stores latent state transition for s_k, a
        self.Rsa = {}  # stores R values for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)
        self.Vs = {}   # stores valid moves at the ROOT node.

    def clear_tree(self) -> None:
        """ Clear all statistics stored in the current search tree """
        self.Qsa, self.Ssa, self.Rsa, self.Nsa, self.Ns, self.Ps, self.Vs = [{} for _ in range(7)]

    def initialize_root(self, state: GameState, trajectory: GameHistory) -> typing.Tuple[typing.Tuple[bytes, tuple],
                                                                                         np.ndarray, float]:
        """
        Embed the provided root state into the MuZero Neural Network. Additionally perform inference for
        this root state and perturb the network prior with Dirichlet noise. As we have access to the game at
        this state, we mask the initial prior with the legal moves at this state.
        :param state: GameState Data structure containing the current state of the environment.
        :param trajectory: GameHistory Data structure containing the entire episode trajectory of the agent(s).
        :return: tuple (hash, latent_state, root_value) The hash/ data of the latent state and inferred root-value.
        """
        # Perform initial inference on o_t-l, ... o_t
        o_t = self.game.buildObservation(state)
        stacked_observations = trajectory.stackObservations(self.neural_net.net_args.observation_length, o_t)
        latent_state, pi_0, v_0 = self.neural_net.initial_inference(stacked_observations)

        s_0 = (latent_state.tobytes(), tuple())  # Hashable representation

        # Add Dirichlet Exploration noise
        noise = np.random.dirichlet([self.args.dirichlet_alpha] * len(pi_0))
        self.Ps[s_0] = noise * self.args.exploration_fraction + (1 - self.args.exploration_fraction) * pi_0

        # Mask the prior for illegal moves, and re-normalize accordingly.
        self.Vs[s_0] = self.game.getLegalMoves(state)

        self.Ps[s_0] *= self.Vs[s_0]
        self.Ps[s_0] = self.Ps[s_0] / np.sum(self.Ps[s_0])

        # Sum of visit counts of the edges/ children
        self.Ns[s_0] = 0

        return s_0, latent_state, v_0

    def compute_ucb(self, s: typing.Tuple[bytes, tuple], a: int, exploration_factor: float) -> float:
        """
        Compute the canonical PUCT formula from the MuZero Paper for an edge (s, a) inside the MCTS tree:

            PUCT(s, a) = MinMaxNormalize(Q(s, a)) + P(s, a) * sqrt(visits_parent / (1 + visits_s)) * exploration_factor

        Illegal edges (only at the root state) are returned as zeros. The Q values within the tree are MinMax
        normalized over the accumulated statistics over the current tree search.

        :param s: tuple Hashable key of the current-state inside the MCTS tree.
        :param a: int Action key representing the path to reach the child node from path (s, a)
        :param exploration_factor: float Pre-computed exploration factor from the MuZero PUCT formula.
        :return: float Upper confidence bound with neural network prior
        """
        if s in self.Vs and not self.Vs[s][a]:
            return -np.inf  # Illegal move masked at a root state.

        visit_count = self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
        q_value = self.minmax.normalize(self.Qsa[(s, a)]) if (s, a) in self.Qsa else 0
        c_children = np.max([self.Ns[s], 1e-8])  # Ensure that prior doesn't collapse to 0 if s is new.

        ucb = self.Ps[s][a] * np.sqrt(c_children) / (1 + visit_count) * exploration_factor  # Exploration
        ucb += q_value                                                                      # Exploitation
        return ucb

    def runMCTS(self, state: GameState, trajectory: GameHistory, temp: float = 1.0) -> typing.Tuple[np.ndarray, float]:
        """
        This function performs 'num_MCTS_sims' simulations of MCTS starting from the provided root GameState.

        Before the search we clear any statistics stored inside the tree (transitions, values, and MinMax bounds).
        In this way we ensure that simulation runtime stays relatively constant over multiple calls to this function
        along with more predictable behaviour and numerical conditioning of the neural embeddings.

        Our estimation of the root-value of the MCTS tree search is based on a sample average of each backed-up
        MCTS value. This means that this estimate represents an on-policy estimate V^pi.

        Illegal moves are masked before computing the action probabilities.

        :param state: GameState Data structure containing the current state of the environment.
        :param trajectory: GameHistory Data structure containing the entire episode trajectory of the agent(s).
        :param temp: float Visit count exponentiation factor. A value of 0 = Greedy, +infinity = uniformly random.
        :return: tuple (pi, v) The move probabilities of MCTS and the estimated root-value of the policy.
        """
        # Refresh value bounds and statistics in the tree
        self.minmax.refresh()
        self.clear_tree()

        # Initialize the root variables needed for MCTS.
        s_0, latent_state, v_0 = self.initialize_root(state, trajectory)

        # Aggregate root state value over MCTS back-propagated values. On-policy.
        v_search = sum([self._search(latent_state) for _ in range(self.args.num_MCTS_sims - 1)])
        v = (v_0 + (v_search if self.single_player else -v_search)) / self.args.num_MCTS_sims

        # MCTS Visit count array for each edge 'a' from root node 's_0'.
        counts = np.array([self.Nsa[(s_0, a)] if (s_0, a) in self.Nsa else 0 for a in range(self.action_size)])
        counts = counts * self.Vs[s_0]  # Mask illegal moves.

        if temp == 0:  # Greedy selection. One hot encode the most visited paths (uniformly random break ties).
            move_probabilities = np.zeros(len(counts))
            move_probabilities[np.argmax(counts + np.random.randn(len(counts)) * 1e-8)] = 1
        else:
            counts = np.power(counts, 1. / temp)
            move_probabilities = counts / np.sum(counts)

        return move_probabilities, v

    def _search(self, latent_state: np.ndarray, path: typing.Tuple[int, ...] = tuple()) -> float:
        """
        Recursively unroll the MuZero Neural Networks with paths guided by the PUCT formula.

        Selection chooses an action for expanding/ traversing the edge (s, a) within the tree search.
        The exploration_factor for the PUCT formula is computed within this function for efficiency:

            exploration_factor = c1 * log(visits_s + c2 + 1) - log(c2)

        If an edge is expanded, we recurrently unroll the dynamics model one step (with action a) and infer
        the new latent-state, reward, move probabilities, and state value. If an edge is traversed, we simply
        look up earlier inferred values from the class dictionaries.

        During backup we update the current value estimates of an edge Q(s, a) using an average, we additionally
        update the MinMax statistics to get reward/ value boundaries for the PUCT formula. Note that backed-up
        values get discounted for gamma < 1. For adversarial games, we negate the backed up value G_k at each backup.

        :param latent_state: np.ndarray Numerical prediction of the state by the encoder/ dynamics model.
        :param path: tuple of integers representing the tree search-path of the current function call.
        :return: float The backed-up discounted/ Monte-Carlo returns (dependent on gamma) of the tree search.
        :raises RecursionError: When cycles occur within the search path, the search can get stuck *ad infinitum*.
        """
        s_k = (latent_state.tobytes(), path)  # Hashable representation.

        ### SELECTION
        # pick the action with the highest upper confidence bound
        exploration_factor = self.args.c1 + np.log(self.Ns[s_k] + self.args.c2 + 1) - np.log(self.args.c2)
        confidence_bounds = [self.compute_ucb(s_k, a, exploration_factor) for a in range(self.action_size)]
        a = np.argmax(confidence_bounds).item()

        if (s_k, a) not in self.Ssa:  ### ROLLOUT
            # Perform a forward pass in the dynamics function.
            reward, next_latent, prior, value = self.neural_net.recurrent_inference(latent_state, a)
            s_k_next = (next_latent.tobytes(), path + (a, ))              # Hashable representation.

            self.Rsa[(s_k, a)], self.Ssa[(s_k, a)] = reward, next_latent  # Current depth statistics
            self.Ps[s_k_next], self.Ns[s_k_next] = prior, 0               # Next depth statistics
            value = value if self.single_player else -value               # Alternate value perspective for adversary.

        else:  ### EXPANSION
            value = self._search(self.Ssa[(s_k, a)], path + (a, ))        # 1-step look ahead state value

        ### BACKUP
        gk = self.Rsa[(s_k, a)] + self.args.gamma * value                 # (Discounted) Value of the current node

        if (s_k, a) in self.Qsa:
            self.Qsa[(s_k, a)] = (self.Nsa[(s_k, a)] * self.Qsa[(s_k, a)] + gk) / (self.Nsa[(s_k, a)] + 1)
            self.Nsa[(s_k, a)] += 1
        else:
            self.Qsa[(s_k, a)] = gk
            self.Nsa[(s_k, a)] = 1

        self.minmax.update(self.Qsa[(s_k, a)])
        self.Ns[s_k] += 1

        return gk if self.single_player else -gk
