"""

"""
import typing
import os
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from utils import DotDict
from utils.loss_utils import scalar_loss, scale_gradient
from utils.debugging import MuZeroMonitor


class MuZeroNeuralNet(ABC):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.
    """

    def __init__(self, game, net_args: DotDict, builder: typing.Callable) -> None:
        """

        :param game:
        :param net_args:
        :param builder:
        """
        self.fit_rewards = (game.n_players == 1)
        self.net_args = net_args
        self.neural_net = builder(game, net_args)
        self.monitor = MuZeroMonitor(self)

        self.optimizer = tf.optimizers.Adam(self.net_args.lr)
        self.steps = 0

    def loss_function(self, observations: tf.Tensor, actions: tf.Tensor, target_vs: tf.Tensor, target_rs: tf.Tensor,
                      target_pis: tf.Tensor, sample_weights: tf.Tensor) -> tf.function:
        """

        :param observations: Shape (batch_size, observation_dimensions)
        :param actions: Shape (batch_size, K, action_size)
        :param target_vs:
        :param target_rs:
        :param target_pis:
        :param sample_weights:
        :return:
        """
        @tf.function
        def loss() -> tf.Tensor:
            """
            Defines the computation graph for computing the loss of a MuZero model given data.

            :return: tf.Tensor with value being the total loss of the MuZero model given the data.
            """
            total_loss = tf.constant(0, dtype=tf.float32)

            # Root inference. Collect predictions of the form: [w_i / K, v, r, pi] for each forward step k = 0...K
            s, pi_0, v_0 = self.neural_net.forward(observations)
            predictions = [(sample_weights, v_0, None, pi_0)]

            for k in range(actions.shape[1]):
                r, s, pi, v = self.neural_net.recurrent([s, actions[:, k, :]])
                predictions.append((tf.divide(sample_weights, len(actions)), v, r, pi))

                s = scale_gradient(s, 1/2)  # Scale the gradient at the start of the dynamics function by 1/2

            # Perform loss computation
            for k in range(len(predictions)):  # Length = 1 + K (root + hypothetical forward steps)
                loss_scale, vs, rs, pis = predictions[k]
                t_vs, t_rs, t_pis = target_vs[k, ...], target_rs[k, ...], target_pis[k, ...]

                r_loss = scalar_loss(rs, t_rs) if (k > 0 and self.fit_rewards) else tf.constant(0, dtype=tf.float32)
                v_loss = scalar_loss(vs, t_vs)
                pi_loss = scalar_loss(pis, t_pis)

                step_loss = r_loss + v_loss + pi_loss
                total_loss += tf.reduce_sum(scale_gradient(step_loss, loss_scale))  # loss_scale includes a 1 / N term.

                # Logging loss of each unrolled head.
                self.monitor.log_recurrent_losses(k, loss_scale, v_loss, r_loss, pi_loss)

            # Penalize magnitude of weights using l2 norm
            l2_norm = tf.reduce_sum([tf.nn.l2_loss(x) for x in self.get_variables()])
            total_loss += self.net_args.l2 * l2_norm

            self.monitor.log(total_loss, "total loss")
            self.monitor.log(l2_norm, "l2 norm")

            return total_loss
        return loss

    @abstractmethod
    def train(self, examples: typing.List) -> None:
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples of the form (observation_trajectory,
                      action_trajectory, targets, loss_scale). Here targets is another
                      tuple comprised of the trajectories of (v, r, pi).
        """
        pass

    @abstractmethod
    def get_variables(self) -> typing.List:
        """
        Yield a list of all trainable variables within the model

        Returns:
            variable_list: A list of all tf.Variables within the entire MuZero model.
        """
        pass

    @abstractmethod
    def initial_inference(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float]:
        """
        Combines the prediction and representation models into one call. This reduces
        overhead and results in a significant speed up.

        Input:
            observations: A game specific (stacked) tensor of observations of the environment at step t: o_t.

        Returns:
            s_(0): The root 'latent_state' produced by the representation function
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float that gives the value of the current board
        """
        pass

    @abstractmethod
    def recurrent_inference(self, latent_state: np.ndarray, action: int) -> typing.Tuple[float, np.ndarray,
                                                                                         np.ndarray, float]:
        """
        Combines the prediction and dynamics models into one call. This reduces
        overhead and results in a significant speed up.

        Input:
            latent_state: A neural encoding of the environment at step k: s_k.
            action: A (encoded) action to perform on the latent state

        Returns:
            r: The immediate predicted reward of the environment.
            s_(k+1): A new 'latent_state' resulting from performing the 'action' in
                the latent_state.
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize.
            v: a float that gives the value of the current board.
        """
        pass

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
