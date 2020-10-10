"""

"""
from dataclasses import dataclass, field
import typing

import numpy as np


@dataclass
class GameHistory:
    """
    Data container for keeping track of game trajectories.
    """
    observations: list = field(default_factory=list)
    players: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    probabilities: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    search_returns: list = field(default_factory=list)
    observed_returns: list = field(default_factory=list)

    def __len__(self) -> int:
        """Get length of current stored trajectory"""
        return len(self.observations)

    def capture(self, observation: np.ndarray, action: int, player: int, pi: np.ndarray, r: float, v: float) -> None:
        """Take a snapshot of the current state of the environment and the search results"""
        self.observations.append(observation)
        self.actions.append(action)
        self.players.append(player)
        self.probabilities.append(pi)
        self.rewards.append(r)
        self.search_returns.append(v)
        self.observed_returns.append(None)

    def terminate(self, observation: np.ndarray, player: int, z) -> None:
        """Take a snapshot of the terminal state of the environment"""
        self.observations.append(observation)
        self.actions.append(-1)
        self.players.append(player)
        self.probabilities.append(None)
        self.rewards.append(0)
        self.search_returns.append(None)
        self.observed_returns.append(z)

    def refresh(self) -> None:
        """Clear all statistics within the class"""
        all(x.clear() for x in vars(self).values())

    def compute_returns(self, gamma: float = 1, n: typing.Optional[int] = None) -> None:
        """Computes the n-step returns assuming that the last recorded snapshot was a terminal state"""
        if n is None:
            # Boardgames
            for i in reversed(range(len(self) - 1)):
                self.observed_returns[i] = -self.observed_returns[i + 1]
        else:
            # General MDPs. Symbols follow notation from the paper.
            for t in range(len(self)):
                horizon = np.min([t + n, len(self)])
                discounted_rewards = [np.pow(gamma, k - t) * self.rewards[k] for k in range(t, horizon)]
                bootstrap = np.pow(gamma, horizon - t) * self.search_returns[horizon]
                self.observed_returns[t] = np.sum(discounted_rewards) + bootstrap

    def stackObservations(self, length: int, current_observation: typing.Optional[np.ndarray] = None,
                          t: typing.Optional[int] = None) -> np.ndarray:
        """Stack the most recent 'length' elements from the observation list along the end of the observation axis"""
        if length <= 1:
            return current_observation if current_observation is not None else self.observations[-1]

        if t is None:
            t = len(self)

        if current_observation is None:
            current_observation = self.observations[t]
            t = np.max([0, t - 1])

        # Get a trajectory of states of 'length' most recent observations until time-point t.
        trajectory = self.observations[:t][-(length - 1):] + [current_observation]

        if len(trajectory) < length:
            prefix = [np.zeros_like(current_observation) for _ in range(length - len(trajectory))]
            trajectory = prefix + trajectory
        # TODO: Functionality check/ unit testing
        return np.concatenate(trajectory, axis=-1)  # Concatenate along channel dimension.


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, minimum_reward: typing.Optional[float] = None,
                 maximum_reward: typing.Optional[float] = None) -> None:
        """

        :param minimum_reward:
        :param maximum_reward:
        """
        self.default_max = self.maximum = maximum_reward if maximum_reward is not None else -np.inf
        self.default_min = self.minimum = minimum_reward if minimum_reward is not None else np.inf

    def refresh(self) -> None:
        """

        :return:
        """
        self.maximum = self.default_max
        self.minimum = self.default_min

    def update(self, value: float) -> None:
        """

        :param value:
        :return:
        """
        self.maximum = np.max([self.maximum, value])
        self.minimum = np.min([self.minimum, value])

    def normalize(self, value: float) -> float:
        """

        :param value:
        :return:
        """
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def sample_batch(list_of_histories: typing.List[GameHistory], n: int, prioritize: bool = False, alpha: float = 1.0,
                 beta: float = 1.0) -> typing.Tuple[typing.List[typing.Tuple[int, int]], typing.List[float]]:
    """
    Generate a sample specification from the list of GameHistory object using uniform or prioritized sampling.
    Along with the generated indices, for each sample/ index a scalar is returned for the loss function during
    backpropagation. For uniform sampling this is simply w_i = 1 / N (Mean) for prioritized sampling this is
    adjusted to

        w_i = (1/ (N * P_i))^beta,

    where P_i is the priority of sample/ index i, defined as

        P_i = (p_i)^alpha / sum_k (p_k)^alpha, with p_i = |v_i - z_i|,

    and v_i being the MCTS search result and z_i being the observed n-step return.
    The w_i is the Importance Sampling ratio and accounts for some of the sampling bias.

    :param list_of_histories: List of GameHistory objects to sample indices from.
    :param n: int Number of samples to generate == batch_size.
    :param prioritize: bool Whether to use prioritized sampling
    :param alpha: float Exponentiation factor for computing priorities, high = uniform, low = greedy
    :param beta: float Exponentiation factor for the Importance Sampling ratio.
    :return: List of tuples indicating a sample, the first index in the tuple specifies which GameHistory object
             within list_of_histories is chosen and the second index specifies the time point within that GameHistory.
             List of scalars containing either the Importance Sampling ratio or 1 / N to scale the network loss with.
    """
    lengths = list(map(len, list_of_histories))   # Map the trajectory length of each Game

    sampling_probability = None                   # None defaults to uniform in np.random.choice
    sample_weight = np.ones(np.sum(lengths)) / n  # 1 / N. Uniform weight update strength over batch.

    if prioritize or alpha == 0:
        errors = np.array([np.abs(h.search_returns[i] - h.observed_returns[i])
                           for h in list_of_histories for i in range(len(h))])

        mass = np.pow(errors, alpha)
        sampling_probability = mass / np.sum(mass)

        # Adjust weight update strength proportionally to IS-ratio to account for sampling bias.
        sample_weight = np.power(n * sampling_probability, -beta)

    # Sample with prioritized / uniform probabilities sample indices over the flattened list of GameHistory objects.
    flat_indices = np.random.choice(a=np.sum(lengths), size=n, replace=False, p=sampling_probability)

    # Map the flat indices to the correct histories and history indices.
    history_index_borders = np.cumsum([0] + lengths)
    history_indices = [(np.sum(i >= history_index_borders), i) for i in flat_indices]

    # Of the form [(history_i, t), ...] \equiv history_it
    sample_coordinates = [(h_i - 1, i - history_index_borders[h_i - 1]) for h_i, i in history_indices]
    # Extract the corresponding IS loss scalars for each sample (or simply N x 1 / N if non-prioritized)
    sample_weights = [sample_weight[lengths[c[0]] + c[1]] for c in sample_coordinates]

    return sample_coordinates, sample_weights
