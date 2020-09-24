import numpy as np

EPS = 1e-8


class MuZeroMCTS:
    """
    This class handles the MCTS tree.
    """

    class MinMaxStats(object):
        """A class that holds the min-max values of the tree."""
        def __init__(self, minimum_reward=None, maximum_reward=None):
            self.default_max = self.maximum = maximum_reward if maximum_reward is not None else -np.inf
            self.default_min = self.minimum = minimum_reward if minimum_reward is not None else np.inf

        def refresh(self):
            self.maximum = self.default_max
            self.minimum = self.default_min

        def update(self, value):
            self.maximum = np.max([self.maximum, value])
            self.minimum = np.min([self.minimum, value])

        def normalize(self, value):
            if self.maximum > self.minimum:
                # We normalize only when we have set the maximum and minimum values.
                return (value - self.minimum) / (self.maximum - self.minimum)
            return value

    def __init__(self, game, neural_net, args):
        self.game = game
        self.neural_net = neural_net
        self.args = args

        # Gets reinitialized at every search
        self.minmax = self.MinMaxStats(self.args.minimum_reward, self.args.maximum_reward)

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Ssa = {}  # stores latent state transition for s_k, a
        self.Rsa = {}  # stores R values for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)

    def turn_indicator(self, counter):
        if self.args.zerosum:
            return 1 if counter % 2 == 0 else -1
        return 1

    def add_exploration_noise(self, s):
        noise = np.random.dirichlet([self.args.dirichlet_alpha] * len(self.Ps[s]))
        self.Ps[s] = noise * self.args.exploration_fraction + (1 - self.args.exploration_fraction) * self.Ps[s]

    def compute_ucb(self, s, a, exploration_factor):
        visit_count = self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
        q_value = self.Qsa[(s, a)] if (s, a) in self.Qsa else 0

        ucb = self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + visit_count) * exploration_factor  # Exploration
        ucb += self.minmax.normalize(q_value)                                               # Exploitation
        return ucb

    def getActionProb(self, observations, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        a history (array) of past observations.

        Returns:
            move_probabilities: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        latent_state = self.neural_net.encode(observations)
        s = latent_state.tostring()  # Hashable representation

        # Refresh value bounds in the tree
        self.minmax.refresh()

        self.search(latent_state, max_depth=0, add_exploration_noise=True)  # Add noise on the first search
        for i in range(self.args.numMCTSSims - 1):
            self.search(latent_state, max_depth=(i+1))

        counts = np.array([self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())])

        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            sample = np.random.choice(best_actions)
            move_probabilities = [0] * len(counts)
            move_probabilities[sample] = 1
            return move_probabilities

        counts = np.power(counts, 1. / temp)
        move_probabilities = counts / np.sum(counts)
        return move_probabilities.tolist()

    def search(self, latent_state, max_depth, count=0, add_exploration_noise=False):
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

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s = latent_state.tostring()  # Hashable representation.

        ### ROLLOUT
        if s not in self.Ps or count == max_depth:
            # leaf node
            self.Ps[s], v = self.neural_net.predict(latent_state)

            if add_exploration_noise:  # Add Dirichlet noise to the root prior.
                self.add_exploration_noise(s)

            self.Ps[s] /= np.sum(self.Ps[s])
            self.Ns[s] = 0
            return self.turn_indicator(count) * v

        ### SELECTION
        # pick the action with the highest upper confidence bound
        exploration_factor = self.args.c1 + np.log(self.Ns[s] + self.args.c2 + 1) - np.log(self.args.c2)
        confidence_bounds = [self.compute_ucb(s, a, exploration_factor) for a in range(self.game.getActionSize() - 1)]
        a = np.argmax(confidence_bounds)

        ### EXPANSION
        # Perform a forward pass using the dynamics function (unless already known in the transition table)
        if (s, a) not in self.Ssa:
            self.Rsa[(s, a)], self.Ssa[(s, a)] = self.neural_net.forward(latent_state, a)

        v = self.search(self.Ssa[(s, a)], max_depth, count=(count+1))  # 1-step look ahead state value
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
