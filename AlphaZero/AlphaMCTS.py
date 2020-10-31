import typing

import numpy as np

from AlphaZero.AlphaNeuralNet import AlphaZeroNeuralNet
from utils import DotDict
from utils.selfplay_utils import MinMaxStats

EPS = 1e-8


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, neural_net: AlphaZeroNeuralNet, args: DotDict) -> None:
        self.game = game
        self.neural_net = neural_net
        self.args = args

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
        self.Qsa, self.Ssa, self.Rsa, self.Nsa, self.Ns, self.Ps, self.Es, self.Vs = [{} for _ in range(8)]

    def initialize_root(self, state) -> typing.Tuple[bytes, float]:
        o_t = self.game.buildObservation(state, 1)
        s_0 = self.game.getHash(state)

        # Initialize root of the tree
        pi_0, v_0 = self.neural_net.predict(o_t)

        # Add Dirichlet Exploration noise
        noise = np.random.dirichlet([self.args.dirichlet_alpha] * len(pi_0))
        self.Ps[s_0] = noise * self.args.exploration_fraction + (1 - self.args.exploration_fraction) * pi_0

        # Mask the prior for illegal moves, and re-normalize accordingly.
        self.Ps[s_0] *= self.game.getLegalMoves(state, 1)
        self.Ps[s_0] = self.Ps[s_0] / np.sum(self.Ps[s_0])

        # Sum of visit counts of the edges/ children and legal moves.
        self.Ns[s_0] = 0
        self.Vs[s_0] = self.game.getLegalMoves(state, 1)

        return s_0, v_0

    def compute_ucb(self, s: bytes, a: int, exploration_factor: float) -> float:
        # Compute the PUCT formula
        visit_count = self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
        q_value = self.minmax.normalize(self.Qsa[(s, a)]) if (s, a) in self.Qsa else 0
        c_children = np.max([self.Ns[s], 1e-8])  # Ensure that prior doesn't collapse to 0 if s is new.

        ucb = self.Ps[s][a] * np.sqrt(c_children) / (1 + visit_count) * exploration_factor  # Exploration
        ucb += q_value                                                                      # Exploitation
        return ucb

    def runMCTS(self, state, temp: int = 1) -> typing.Tuple[np.ndarray, float]:
        """
        This function performs numMCTSSims simulations of MCTS starting from
        a history (array) of past observations.

        Returns:
            move_probabilities: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s_0,a)]**(1./temp)
            v: (float) Estimated value of the root state.
        """
        s_0, v_0 = self.initialize_root(state)

        # Refresh value bounds in the tree
        self.minmax.refresh()

        # Aggregate root state value over MCTS back-propagated values
        v_search = sum([self._search(state) for _ in range(self.args.numMCTSSims)])
        v = (v_0 + (-v_search if self.game.n_players > 1 else v_search)) / self.args.numMCTSSims

        # MCTS Visit count array for each edge 'a' from root node 's_0'.
        counts = np.array([self.Nsa[(s_0, a)] if (s_0, a) in self.Nsa else 0 for a in range(self.game.getActionSize())])

        if temp == 0:  # Greedy selection. One hot encode the most visited paths (randomly break ties).
            move_probabilities = np.zeros(len(counts))
            move_probabilities[np.argmax(counts + np.random.randn(len(counts)) * 1e-8)] = 1
        else:
            counts = np.power(counts, 1. / temp)
            move_probabilities = counts / np.sum(counts)

        return move_probabilities, v

    def _search(self, state, path: typing.Tuple[int, ...] = tuple()) -> float:
        """

        """
        s = self.game.getHash(state)

        ### SELECTION
        # pick the action with the highest upper confidence bound
        exploration_factor = self.args.c1 + np.log(self.Ns[s] + self.args.c2 + 1) - np.log(self.args.c2)
        confidence_bounds = [self.compute_ucb(s, a, exploration_factor) for a in range(self.game.getActionSize())]
        a = np.argmax(self.Vs[s] * np.asarray(confidence_bounds)).item()  # Get argmax as scalar

        if (s, a) not in self.Ssa:  ### ROLLOUT
            next_state, reward, next_player = self.game.getNextState(state, a, 1, clone=True)
            next_state = self.game.getCanonicalForm(next_state, next_player)

            if next_state not in self.Es:
                self.Es[next_state] = self.game.getGameEnded(next_state, 1)

            if self.Es[next_state]:  # Leaf node
                value = 0 if self.game.n_players == 1 else -1
                self.Rsa[(s, a)] = 0
            else:
                # Build network input for inference
                o_t = self.game.buildObservation(next_state, 1)

                prior, value = self.neural_net.predict(o_t)
                s_next = self.game.getHash(next_state)

                # Current depth statistics
                self.Rsa[(s, a)], self.Ssa[(s, a)] = reward, (next_state, next_player)
                # Next depth statistics
                self.Ps[s_next], self.Ns[s_next], self.Vs[s_next] = prior, 0, self.game.getLegalMoves(next_state, 1)
                value = value if self.game.n_players == 1 else -value  # Alternate value perspective for adversary.

        else:  ### EXPANSION
            value = self._search(self.Ssa[(s, a)][0], path + (a,))  # 1-step look ahead state value

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

        return -gk if self.game.n_players > 1 else gk
