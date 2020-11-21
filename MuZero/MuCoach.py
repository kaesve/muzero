"""
Implements the abstract Coach class for defining the data sampling procedures for MuZero neural network training.

Notes:
 - Base implementation done.
 - Documentation 15/11/2020
"""
import typing
from datetime import datetime

import numpy as np
import tensorflow as tf

from Coach import Coach
from Agents import DefaultMuZeroPlayer
from MuZero.MuMCTS import MuZeroMCTS
from utils import DotDict
from utils.selfplay_utils import GameHistory, sample_batch


class MuZeroCoach(Coach):
    """
    Implement base Coach class to define proper data-batch sampling procedures and logging objects.
    """

    def __init__(self, game, neural_net, args: DotDict, run_name: typing.Optional[str] = None) -> None:
        """
        Initialize the class for self-play. This inherited method initializes tensorboard logging and defines
        helper variables for data batch sampling.

        The super class is initialized with the proper search engine and agent-interface. (MuZeroMCTS, MuZeroPlayer)

        :param game: Game Implementation of Game class for environment logic.
        :param neural_net: MuNeuralNet Implementation of MuNeuralNet class for inference.
        :param args: DotDict Data structure containing parameters for self-play.
        :param run_name: str Optionally provide a run-name for the TensorBoard log-files. Default is current datetime.
        """
        super().__init__(game, neural_net, args, MuZeroMCTS, DefaultMuZeroPlayer)

        # Initialize tensorboard logging.
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_dir = f"out/logs/MuZero/{self.neural_net.architecture}/" + run_name
        self.file_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        self.file_writer.set_as_default()

        # Define helper variables.
        self.return_forward_observations = (neural_net.net_args.dynamics_penalty > 0 or args.latent_decoder)
        self.observation_stack_length = neural_net.net_args.observation_length

    def buildHypotheticalSteps(self, history: GameHistory, t: int, k: int) -> \
            typing.Tuple[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Sample/ extrapolate a sequence of targets for unrolling/ fitting the MuZero neural network.

        This sequence consists of the actions performed at time t until t + k - 1. These are used for unrolling the
        dynamics model. For extrapolating beyond terminal states we adopt an uniform policy over the entire action
        space to ensure that the model learns to generalize over the actions when encountering terminal states.

        The move-probabilities, value, and reward predictions are sampled from t until t + k. Note that the reward
        at the first index is not used for weight optimization as the initial call to the model does not predict
        rewards. For extrapolating beyond terminal states we repeat a zero vector for the move-probabilities and
        zeros for the reward and value targets seeing as a terminated environment does not provide rewards. The
        zero vector for the move-probabilities is used to define an improper probability distribution. The loss
        function can then infer that the episode ended, and distribute gradient accordingly.

        Empirically we observed that extrapolating an uniform move-policy for the move-probability vector results
        in slower and more unstable learning as we're feeding wrong data to the neural networks. We found that not
        distributing any gradient at all to these extrapolated steps resulted in the best learning.

        :param history: GameHistory Sampled data structure containing all statistics/ observations of a finished game.
        :param t: int The sampled index to generate the targets at.
        :param k: int The number of unrolling steps to perform/ length of the dynamics model target sequence.
        :return: Tuple of (actions, targets, future_inputs) that the neural network needs for optimization
        """
        # One hot encode actions.
        actions = history.actions[t:t+k]
        a_truncation = k - len(actions)
        if a_truncation > 0:  # Uniform policy when unrolling beyond terminal states.
            actions += np.random.randint(self.game.getActionSize(), size=a_truncation).tolist()

        enc_actions = np.zeros([k, self.game.getActionSize()])
        enc_actions[np.arange(len(actions)), actions] = 1

        # Value targets.
        pis = history.probabilities[t:t+k+1]
        vs = history.observed_returns[t:t+k+1]
        rewards = history.rewards[t:t+k+1]

        # Handle truncations > 0 due to terminal states. Treat last state as absorbing state
        t_truncation = (k + 1) - len(pis)  # Target truncation due to terminal state
        if t_truncation > 0:
            pis += [np.zeros_like(pis[-1])] * t_truncation  # Zero vector
            rewards += [0] * t_truncation                   # = 0
            vs += [0] * t_truncation                        # = 0

        # If specified, also sample/ extrapolate future observations. Otherwise return an empty array.
        obs_trajectory = []
        if self.return_forward_observations:
            obs_trajectory = [history.stackObservations(self.observation_stack_length, t=t+i+1) for i in range(k)]

        # (Actions, Targets, Observations)
        return enc_actions, (np.asarray(vs), np.asarray(rewards), np.asarray(pis)), obs_trajectory

    def sampleBatch(self, histories: typing.List[GameHistory]) -> typing.List:
        """
        Construct a batch of data-targets for gradient optimization of the MuZero neural network.

        The procedure samples a list of game and inside-game coordinates of length 'batch_size'. This is done either
        uniformly or with prioritized sampling. Using this list of coordinates, we sample the according games, and
        the according points of times within the game to generate neural network inputs, targets, and sample weights.

        :param histories: List of GameHistory objects. Contains all game-trajectories in the replay-buffer.
        :return: List of training examples: (observations, actions, targets, forward_observations, sample_weights)
        """
        # Generate coordinates within the replay buffer to sample from. Also generate the loss scale of said samples.
        sample_coordinates, sample_weight = sample_batch(
            list_of_histories=histories, n=self.neural_net.net_args.batch_size, prioritize=self.args.prioritize,
            alpha=self.args.prioritize_alpha, beta=self.args.prioritize_beta)

        # Collect training examples for MuZero: (input, action, (targets), forward_observations, loss_scale)
        examples = [(
            histories[h_i].stackObservations(self.observation_stack_length, t=i),
            *self.buildHypotheticalSteps(histories[h_i], t=i, k=self.args.K),
            loss_scale
        )
            for (h_i, i), loss_scale in zip(sample_coordinates, sample_weight)
        ]

        return examples
