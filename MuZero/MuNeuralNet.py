"""
Defines the base structure of a neural network for MuZero. Provides a generic implementation
for performing loss computation and recurrent unrolling of the network given data.

Notes:
 - Base implementation done.
 - Documentation 14/11/2020
"""
import typing
import os
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from utils import DotDict
from utils.loss_utils import scalar_loss, scale_gradient, safe_l2norm
from utils.debugging import MuZeroMonitor


class MuZeroNeuralNet(ABC):
    """
    This class specifies the base NeuralNet class. To define your own neural network, subclass
    this class and implement the abstract functions below. The unroll and loss_function methods
    should be generic enough to work with any canonical MuZero architecture.
    """

    def __init__(self, game, net_args: DotDict, builder: typing.Callable) -> None:
        """
        Initialize base MuZero Neural Network. Contains all requisite logic to work with any
        MuZero network and environment.
        :param game: Implementation of base Game class for environment logic.
        :param net_args: DotDict Data structure that contains all neural network arguments as object attributes.
        :param builder: Function that takes the game and network arguments as parameters and returns a tf.keras.Model
        :raises: NotImplementedError if invalid optimization method is specified in the provided .json configuration.
        """
        self.fit_rewards = (game.n_players == 1)
        self.net_args = net_args
        self.neural_net = builder(game, net_args)
        self.monitor = MuZeroMonitor(self)
        self.steps = 0

        # Select parameter optimizer from config.
        if self.net_args.optimizer.method == "adam":
            self.optimizer = tf.optimizers.Adam(lr=self.net_args.optimizer.lr_init)
        elif self.net_args.optimizer.method == "sgd":
            self.optimizer = tf.optimizers.SGD(lr=self.net_args.optimizer.lr_init,
                                               momentum=self.net_args.optimizer.momentum)
        else:
            raise NotImplementedError(f"Optimization method {self.net_args.optimizer.method} not implemented...")

    @tf.function
    def unroll(self, observations: tf.Tensor, actions: tf.Tensor) -> typing.List[typing.Tuple[tf.Tensor, tf.Tensor,
                                                                                              tf.Tensor, tf.Tensor]]:
        """
        Build up a computation graph that collects output tensors from recurrently unrolling the MuZero model.

        After each recurrent unrolling, the graph contains a layer that halves reverse differentiated gradients.

        :param observations: tf.Tensor in R^(batch_size x width x height x (depth * time))
        :param actions: tf.Tensor consisting of one-hot-encoded actions in {0, 1}^(batch_size x K x |action_space|)
        :return: List of tuples containing the value predictions and loss-scale for each unrolled step.
        """
        # Root inference. Collect predictions of the form: [w_i / K, v, r, pi] for each forward step k = 0...K
        s, pi_0, v_0 = self.neural_net.forward(observations)

        # Note: Root can be a terminal state. Loss scale for the root head is 1.0 instead of 1 / K.
        predictions = [(1.0, v_0, 0, pi_0)]
        for k in range(actions.shape[1]):
            r, s, pi, v = self.neural_net.recurrent([s, actions[:, k, :]])
            predictions.append((1.0 / actions.shape[1], v, r, pi))

            # Scale the gradient at the start of the dynamics function by 1/2
            s = scale_gradient(s, 0.5)

        return predictions

    @tf.function
    def loss_function(self, observations: tf.Tensor, actions: tf.Tensor, target_vs: tf.Tensor, target_rs: tf.Tensor,
                      target_pis: tf.Tensor, sample_weights: tf.Tensor) -> typing.Tuple[tf.Tensor, typing.List]:
        """
        Defines the computation graph for computing the loss of a MuZero model given data.

        The function recurrently unrolls the MuZero neural network based on data trajectories.
        From the collected output tensors, this function aggregates the loss for each prediction head for
        each unrolled time step k = 0, 1, ..., K.

        We expect target_pis/ MCTS probability vectors extrapolated beyond terminal states (i.e., no valid search
        statistics) to be a zero-vector. This is important as we infer the unrolling beyond terminal states by
        summing the MCTS probability vectors assuming that they should define proper distributions.

        For unrolled states beyond terminal environment states, we cancel the gradient for the probability vector.
        We keep gradients for the value and reward prediction, so that they learn to recognize terminal states
        during MCTS search. Note that the root state could be provided as a terminal state, this would mean that
        the probability vector head would receive zero gradient for the entire unrolling.

        :param observations: tf.Tensor in R^(batch_size x width x height x (depth * time)). Stacked state observations.
        :param actions: tf.Tensor in {0, 1}^(batch_size x K x |action_space|). One-hot encoded actions for unrolling.
        :param target_vs: tf.Tensor either in [0,1] or R with dimensions (K x batch_size x support_size)
        :param target_rs: tf.Tensor either in [0,1] or R with dimensions (K x batch_size x support_size)
        :param target_pis: tf.Tensor either in [0,1] or R with dimensions (K x batch_size x |action_space|)
        :param sample_weights: tf.Tensor in [0, 1]^(batch_size). Of the form (batch_size * priority) ^ (-beta)
        :return: tuple of a tf.Tensor and a list of tf.Tensors containing the total loss and piecewise losses.
        :see: MuNeuralNet.unroll
        """
        loss_monitor = []  # Collect losses for logging.

        # Sum over target probabilities. Absorbing states should have a zero sum --> leaf node.
        absorb_k = 1.0 - tf.reduce_sum(target_pis, axis=-1)

        # Root inference. Collect predictions of the form: [w_i / K, v, r, pi] for each forward step k = 0...K
        predictions = self.unroll(observations, actions)

        # Perform loss computation for each unrolling step.
        total_loss = tf.constant(0.0, dtype=tf.float32)
        for k in range(len(predictions)):  # Length = 1 + K (root + hypothetical forward steps)
            loss_scale, vs, rs, pis = predictions[k]
            t_vs, t_rs, t_pis = target_vs[k, ...], target_rs[k, ...], target_pis[k, ...]

            absorb = absorb_k[k, :]

            # Calculate losses per head. Cancel gradients in prior for absorbing states, keep gradients for r and v.
            r_loss = scalar_loss(rs, t_rs) if (k > 0 and self.fit_rewards) else tf.constant(0, dtype=tf.float32)
            v_loss = scalar_loss(vs, t_vs)
            pi_loss = scalar_loss(pis, t_pis) * (1.0 - absorb)

            step_loss = scale_gradient(r_loss + v_loss + pi_loss, loss_scale * sample_weights)
            total_loss += tf.reduce_sum(step_loss)  # Actually averages over batch : see sample_weights.

            # Logging
            loss_monitor.append((v_loss, r_loss, pi_loss, absorb))

        # Penalize magnitude of weights using l2 norm
        l2_norm = tf.reduce_sum([safe_l2norm(x) for x in self.get_variables()])
        total_loss += self.net_args.l2 * l2_norm

        return total_loss, loss_monitor

    @abstractmethod
    def train(self, examples: typing.List) -> None:
        """
        This function trains the neural network with data gathered from self-play.

        :param examples: a list of training examples of the form:
                         (observation_trajectories, action_trajectories, targets, loss_scales).
        """

    @abstractmethod
    def get_variables(self) -> typing.List:
        """
        Yield a list of all trainable variables within the model

        :returns: List A list of all tf.Variables within the entire MuZero model.
        """

    @abstractmethod
    def initial_inference(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float]:
        """
        Combines the prediction and representation models into one call. This reduces
        overhead and results in a significant speed up.

        :param observations: A game specific (stacked) tensor of observations of the environment at step t: o_t.

        :returns: A tuple with predictions of the following form:
            s_(0): The root 'latent_state' produced by the representation function
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float that gives the value of the current board
        """

    @abstractmethod
    def recurrent_inference(self, latent_state: np.ndarray, action: int) -> typing.Tuple[float, np.ndarray,
                                                                                         np.ndarray, float]:
        """
        Combines the prediction and dynamics models into one call. This reduces
        overhead and results in a significant speed up.

        :param latent_state: A neural encoding of the environment at step k: s_k.
        :param action: A (encoded) action to perform on the latent state

        :returns: A tuple with predictions of the following form:
            r: The immediate predicted reward of the environment.
            s_(k+1): A new 'latent_state' resulting from performing the 'action' in
                the latent_state.
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize.
            v: a float that gives the value of the current board.
        """

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """ Saves the current neural network (with its parameters) in folder/filename """
        representation_path = os.path.join(folder, 'r_' + filename)
        dynamics_path = os.path.join(folder, 'd_' + filename)
        predictor_path = os.path.join(folder, 'p_' + filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.neural_net.encoder.save_weights(representation_path)
        self.neural_net.dynamics.save_weights(dynamics_path)
        self.neural_net.predictor.save_weights(predictor_path)

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """ Loads parameters of each neural network model from given folder/filename """
        representation_path = os.path.join(folder, 'r_' + filename)
        dynamics_path = os.path.join(folder, 'd_' + filename)
        predictor_path = os.path.join(folder, 'p_' + filename)

        if not os.path.exists(representation_path):
            raise FileNotFoundError(f"No MuZero Representation Model in path {representation_path}")
        if not os.path.exists(dynamics_path):
            raise FileNotFoundError(f"No MuZero Dynamics Model in path {dynamics_path}")
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"No MuZero Predictor Model in path {predictor_path}")

        self.neural_net.encoder.load_weights(representation_path)
        self.neural_net.dynamics.load_weights(dynamics_path)
        self.neural_net.predictor.load_weights(predictor_path)
