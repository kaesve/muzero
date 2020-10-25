"""

"""
import typing

import numpy as np

from MuZero.MuNeuralNet import MuZeroNeuralNet
from utils import DotDict
from utils.selfplay_utils import MinMaxStats

EPS = 1e-8


class MuZeroMCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, neural_net: MuZeroNeuralNet, args: DotDict) -> None:
        """

        :param game:
        :param neural_net:
        :param args:
        """
        self.game = game
        self.neural_net = neural_net
        self.args = args

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
        """Clear all statistics stored in the current search tree"""
        self.Qsa, self.Ssa, self.Rsa, self.Nsa, self.Ns, self.Ps, self.Vs = [{} for _ in range(7)]

    def initialize_root(self, observations: np.ndarray,
                        legal_moves: np.ndarray) -> typing.Tuple[typing.Tuple[bytes, tuple], np.ndarray, float]:
        """

        :param observations:
        :param legal_moves:
        :return:
        """
        latent_state, pi_0, v_0 = self.neural_net.initial_inference(observations)
        s_0 = (latent_state.tobytes(), tuple())  # Hashable representation

        # Add Dirichlet Exploration noise
        noise = np.random.dirichlet([self.args.dirichlet_alpha] * len(pi_0))
        self.Ps[s_0] = noise * self.args.exploration_fraction + (1 - self.args.exploration_fraction) * pi_0

        # Mask the prior for illegal moves, and re-normalize accordingly.
        self.Ps[s_0] *= legal_moves
        self.Ps[s_0] = self.Ps[s_0] / np.sum(self.Ps[s_0])

        # Sum of visit counts of the edges/ children
        self.Ns[s_0] = 0

        return s_0, latent_state, v_0

    def compute_ucb(self, s: typing.Tuple[bytes, tuple], a: int, exploration_factor: float) -> float:
        # Compute the PUCT formula
        visit_count = self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
        q_value = self.minmax.normalize(self.Qsa[(s, a)]) if (s, a) in self.Qsa else 0
        c_children = np.max([self.Ns[s], 1e-8])  # Ensure that prior doesn't collapse to 0 if s is new.

        ucb = self.Ps[s][a] * np.sqrt(c_children) / (1 + visit_count) * exploration_factor  # Exploration
        ucb += q_value                                                                      # Exploitation
        return ucb

    def runMCTS(self, observations: np.ndarray, legal_moves: np.ndarray,
                temp: int = 1) -> typing.Tuple[np.ndarray, float]:
        """
        This function performs numMCTSSims simulations of MCTS starting from
        a history (array) of past observations.

        Returns:
            move_probabilities: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s_0,a)]**(1./temp)
            v: (float) Estimated value of the root state.
        """
        # Get a hashable latent state representation 's_0', the predicted value of the root, and the numerical root s_0.
        s_0, latent_state, v_0 = self.initialize_root(observations, legal_moves)

        # Refresh value bounds in the tree
        self.minmax.refresh()

        # Aggregate root state value over MCTS back-propagated values
        v_search = sum([self._search(latent_state) for _ in range(self.args.numMCTSSims)])
        v = (v_0 + (-v_search if self.game.n_players > 1 else v_search)) / self.args.numMCTSSims

        # MCTS Visit count array for each edge 'a' from root node 's_0'.
        counts = np.array([self.Nsa[(s_0, a)] if (s_0, a) in self.Nsa else 0 for a in range(self.game.getActionSize())])

        if temp == 0:  # Greedy selection. One hot encode the most visited paths (uniformly random break ties).
            move_probabilities = np.zeros(len(counts))
            move_probabilities[np.argmax(counts + np.random.randn(len(counts)) * 1e-8)] = 1
        else:
            counts = np.power(counts, 1. / temp)
            move_probabilities = counts / np.sum(counts)

        return move_probabilities, v

    def _search(self, latent_state: np.ndarray, path: typing.Tuple[int, ...] = tuple()) -> float:
        """

        """
        s_k = (latent_state.tobytes(), path)  # Hashable representation.

        ### SELECTION
        # pick the action with the highest upper confidence bound
        exploration_factor = self.args.c1 + np.log(self.Ns[s_k] + self.args.c2 + 1) - np.log(self.args.c2)
        confidence_bounds = [self.compute_ucb(s_k, a, exploration_factor) for a in range(self.game.getActionSize())]
        a = np.argmax(confidence_bounds).item()  # Get argmax as scalar

        if (s_k, a) not in self.Ssa:  ### ROLLOUT
            # Perform a forward pass in the dynamics function.
            reward, next_latent, prior, value = self.neural_net.recurrent_inference(latent_state, a)
            s_k_next = (next_latent.tobytes(), path + (a, ))  # Hashable representation.

            self.Rsa[(s_k, a)], self.Ssa[(s_k, a)] = reward, next_latent  # Current depth statistics
            self.Ps[s_k_next], self.Ns[s_k_next] = prior, 0               # Next depth statistics
            value = value if self.game.n_players == 1 else -value         # Alternate value perspective for adversary.

        else:  ### EXPANSION
            value = self._search(self.Ssa[(s_k, a)], path + (a, ))        # 1-step look ahead state value

        ### BACKUP
        gk = self.Rsa[(s_k, a)] + self.args.gamma * value   # (Discounted) Value of the current node

        if (s_k, a) in self.Qsa:
            self.Qsa[(s_k, a)] = (self.Nsa[(s_k, a)] * self.Qsa[(s_k, a)] + gk) / (self.Nsa[(s_k, a)] + 1)
            self.Nsa[(s_k, a)] += 1
        else:
            self.Qsa[(s_k, a)] = gk
            self.Nsa[(s_k, a)] = 1

        self.minmax.update(self.Qsa[(s_k, a)])
        self.Ns[s_k] += 1

        return -gk if self.game.n_players > 1 else gk
