import numpy as np

EPS = 1e-8

class MuZeroMCTS:
    """
    This class handles the MCTS tree.
    """

    class MinMaxStats(object):
        """A class that holds the min-max values of the tree."""
        def __init__(self, minimum_reward=None, maximum_reward=None):
            self.maximum = maximum_reward if maximum_reward is not None else -np.inf
            self.minimum = minimum_reward if minimum_reward is not None else np.inf

        def update(self, value):
            self.maximum = max(self.maximum, value)
            self.minimum = min(self.minimum, value)

        def normalize(self, value):
            if self.maximum > self.minimum:
                # We normalize only when we have set the maximum and minimum values.
                return (value - self.minimum) / (self.maximum - self.minimum)
            return value

    def __init__(self, game, neural_net, args):
        self.game = game
        self.neural_net = neural_net
        self.args = args

        self.minmax = self.MinMaxStats(args.minimum_reward, args.maximum_reward)

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
        prior = self.Ps[s]

    def compute_upper_confidence_bound(self, s, a, exploration_factor):
        ucb = self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)]) * exploration_factor  # Exploration
        ucb += self.minmax.normalize(self.Rsa[(s, a)] + self.args.gamma * self.Qsa[(s, a)]
                                     if (s, a) in self.Qsa else 0)                               # Exploitation
        return ucb

    def getActionProb(self, observations, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        a history (array) of past observations.

        Returns:
            move_probabilities: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s_0 = self.neural_net.encode(observations)
        s = s_0.tostring()

        # TODO: Add slight Dirichlet noise to the prior of the nodes during training.
        for i in range(self.args.numMCTSSims):
            self.search(s_0)

        counts = np.array([self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())])

        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            sample = np.random.choice(best_actions)
            move_probabilities = np.zeros(len(counts))
            move_probabilities[sample] = 1
            return move_probabilities

        counts = np.power(counts, 1. / temp)
        move_probabilities = counts / np.sum(counts)
        return move_probabilities

    def search(self, latent_state, count=0):
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
        s = latent_state.tostring()

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.neural_net.predict(latent_state)

            if len(self.Ps) == 0:  # Add Dirichlet noise to the root prior.
                self.add_exploration_noise(s)

            self.Ps[s] /= np.sum(self.Ps[s])
            self.Ns[s] = 0
            return self.turn_indicator(count) * v

        ### SELECTION
        # pick the action with the highest upper confidence bound
        exploration_factor = self.args.c1 + np.log(self.Ns[s] + self.args.c2 + 1) - np.log(self.args.c2)
        confidence_bounds = [self.compute_upper_confidence_bound(s, a, exploration_factor)
                             for a in range(self.game.getActionSize())]
        a = np.argmax(confidence_bounds)

        ### Expansion/ tree traversal
        # Perform a forward pass using the dynamics function (unless already known in the transition table)
        if (s, a) not in self.Ssa:
            self.Rsa[(s, a)], self.Ssa[(s, a)] = self.neural_net.forward(latent_state, a)

        v = self.search(self.Ssa[(s, a)], count + 1)
        gk = self.Rsa[(s, a)] + self.args.gamma * v

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + gk) / (self.Nsa[(s, a)] + 1)
            self.minmax.update(self.Qsa[(s, a)])
        else:
            self.Qsa[(s, a)] = v

        self.Nsa[(s, a)] = 1
        self.Ns[s] += 1

        return self.turn_indicator(count) * gk  # This was changed from -v
