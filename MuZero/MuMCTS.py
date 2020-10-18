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
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Vs = {}  # stores valid moves at the ROOT node.

    def clear_tree(self) -> None:
        """Clear all statistics stored in the current search tree"""
        self.Qsa, self.Ssa, self.Rsa, self.Nsa, self.Ns, self.Ps, self.Vs = [{} for _ in range(7)]

    def turn_indicator(self, counter: int) -> int:
        """

        :param counter:
        :return:
        """
        return 1 if counter % self.game.n_players == 0 else -1

    def modify_root_prior(self, s: tuple) -> None:
        """

        :param s:
        :return:
        """
        # Add Dirichlet Exploration noise
        noise = np.random.dirichlet([self.args.dirichlet_alpha] * len(self.Ps[s]))
        self.Ps[s] = noise * self.args.exploration_fraction + (1 - self.args.exploration_fraction) * self.Ps[s]

        # Mask the prior for illegal moves, and re-normalize accordingly.
        self.Ps[s] *= self.Vs[s]
        self.Ps[s] = self.Ps[s] / np.sum(self.Ps[s])

    def compute_ucb(self, s: tuple, a: int, exploration_factor: float) -> float:
        """

        :param s:
        :param a:
        :param exploration_factor:
        :return:
        """
        visit_count = self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
        q_value = self.Qsa[(s, a)] if (s, a) in self.Qsa else 0
        c_children = np.max([self.Ns[s], 1e-4])  # Ensure that prior doesn't collapse to 0 if s is new.

        ucb = self.Ps[s][a] * np.sqrt(c_children) / (1 + visit_count) * exploration_factor  # Exploration
        ucb += self.minmax.normalize(q_value)                                               # Exploitation
        return ucb

    def runMCTS(self, observations: np.ndarray, legal_moves: np.ndarray,
                temp: int = 1) -> typing.Tuple[np.ndarray, float]:
        """
        This function performs numMCTSSims simulations of MCTS starting from
        a history (array) of past observations. The current state observation must
        be at the front of the observations array! observations[-1] == o_t

        Returns:
            move_probabilities: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
            v: (float) Estimated value of the root state.
        """
        latent_state = self.neural_net.encode(observations)
        s = (latent_state.tobytes(), tuple())  # Hashable representation

        self.clear_tree()
        # Refresh value bounds in the tree
        self.minmax.refresh()
        # Initialize legal moves ONLY at the root.
        self.Vs[s] = legal_moves

        v_sum = 0
        for i in range(self.args.numMCTSSims):
            v_sum += self._search(latent_state, root=(i == 0))
        v = v_sum / self.args.numMCTSSims

        # MCTS Visit count array for each edge 'a' from root node 's'.
        counts = np.array([self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())])

        if temp == 0:  # Greedy selection.
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            sample = np.random.choice(best_actions)
            move_probabilities = np.zeros(len(counts))
            move_probabilities[sample] = 1
            return move_probabilities, v

        counts = np.power(counts, 1. / temp)
        move_probabilities = counts / np.sum(counts)
        return move_probabilities, v

    def _search(self, latent_state: np.ndarray, path: typing.Tuple[int, ...] = tuple(), root: bool = False) -> float:
        """

        """
        s = (latent_state.tobytes(), path)  # Hashable representation.

        ### ROLLOUT
        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.neural_net.predict(latent_state)
            self.Ns[s] = 0

            if root:  # Add Dirichlet noise to the root prior and mask illegal moves.
                self.modify_root_prior(s)

            return self.turn_indicator(len(path)) * v

        ### SELECTION
        # pick the action with the highest upper confidence bound
        exploration_factor = self.args.c1 + np.log(self.Ns[s] + self.args.c2 + 1) - np.log(self.args.c2)
        confidence_bounds = [self.compute_ucb(s, a, exploration_factor) for a in range(self.game.getActionSize())]

        # Only the root node has access to the set of legal actions.
        if s in self.Vs:
            confidence_bounds = np.array(confidence_bounds) * self.Vs[s]
        a = np.argmax(confidence_bounds).item()  # Get argmax as scalar

        ### EXPANSION
        # Perform a forward pass using the dynamics function (unless already known in the transition table)
        if (s, a) not in self.Ssa:
            self.Rsa[(s, a)], self.Ssa[(s, a)] = self.neural_net.forward(latent_state, a)

        v = self._search(self.Ssa[(s, a)], path + (a, ))  # 1-step look ahead state value
        gk = self.Rsa[(s, a)] + self.args.gamma * v   # (Discounted) Value of the current node

        ### BACKUP
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + gk) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = gk
            self.Nsa[(s, a)] = 1

        self.minmax.update(self.Qsa[(s, a)])
        self.Ns[s] += 1

        return self.turn_indicator(len(path)) * gk
