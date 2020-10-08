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
        if self.args.boardgame:
            return 1 if counter % 2 == 0 else -1
        return 1

    def modify_root_prior(self, s: np.ndarray) -> None:
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

    def compute_ucb(self, s: str, a: int, exploration_factor: float) -> float:
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

    def runMCTS(self, observations: np.ndarray, temp: int = 1) -> typing.Tuple[np.ndarray, float]:
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
        s = latent_state.tostring()  # Hashable representation

        # Refresh value bounds in the tree
        self.minmax.refresh()
        # Initialize legal moves ONLY at the root.
        self.Vs[s] = self.game.getLegalMoves(state=observations[-1], player=1)

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

    def _search(self, latent_state: np.ndarray, count: int = 0, root: bool = False) -> float:
        """ TODO: Edit documentation
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        max_depth is included to prevent infinite recursion when the network
        predicts gets stuck in badly conditioned output spaces.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s = latent_state.tostring()  # Hashable representation.

        ### ROLLOUT
        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.neural_net.predict(latent_state)
            self.Ns[s] = 0

            if root:  # Add Dirichlet noise to the root prior and mask illegal moves.
                self.modify_root_prior(s)

            return self.turn_indicator(count) * v

        ### SELECTION
        # pick the action with the highest upper confidence bound
        exploration_factor = self.args.c1 + np.log(self.Ns[s] + self.args.c2 + 1) - np.log(self.args.c2)
        confidence_bounds = [self.compute_ucb(s, a, exploration_factor) for a in range(self.game.getActionSize() - 1)]

        if count > self.args.numMCTSSims + 1:
            print(f'Warning: latent state may have collapsed, defaulting to random search. Recursion depth: {count}')
            confidence_bounds = np.random.uniform(size=len(confidence_bounds))

        # Only the root node has access to the set of legal actions.
        if s in self.Vs:
            confidence_bounds = np.array(confidence_bounds) * self.Vs[s][:-1]  # Omit resignation.
        a = np.argmax(confidence_bounds)

        ### EXPANSION
        # Perform a forward pass using the dynamics function (unless already known in the transition table)
        if (s, a) not in self.Ssa:
            self.Rsa[(s, a)], self.Ssa[(s, a)] = self.neural_net.forward(latent_state, a)

        v = self._search(self.Ssa[(s, a)], count + 1)  # 1-step look ahead state value
        gk = self.Rsa[(s, a)] + self.args.gamma * v   # (Discounted) Value of the current node

        ### BACKUP
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + gk) / (self.Nsa[(s, a)] + 1)
        else:
            self.Qsa[(s, a)] = gk
        self.minmax.update(self.Qsa[(s, a)])
        self.Nsa[(s, a)] = 1
        self.Ns[s] += 1

        return self.turn_indicator(count) * gk  # This was changed from -v
