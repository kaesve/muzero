"""

"""
import typing

import tensorflow as tf
import numpy as np

from utils.storage import DotDict
from utils.loss_utils import scalar_loss, scale_gradient


class MuZeroNeuralNet:
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game, net_args: DotDict, builder: typing.Callable) -> None:
        """

        :param game:
        :param net_args:
        :param builder:
        """
        self.net_args = net_args
        self.neural_net = builder(game, net_args)

        self.optimizer = tf.optimizers.Adam(self.net_args.lr)

    def loss_function(self, observations: tf.Tensor, actions: tf.Tensor, target_vs: tf.Tensor, target_rs: tf.Tensor,
                      target_pis: tf.Tensor, sample_weights: tf.Tensor) -> typing.Callable:
        """

        :param observations:
        :param actions:
        :param target_vs:
        :param target_rs:
        :param target_pis:
        :param sample_weights:
        :return:
        """

        @tf.function
        def loss() -> tf.Tensor:
            """

            :return:
            """
            total_loss = tf.constant(0, dtype=tf.float32)

            # Root inference
            s = self.neural_net.encoder(observations)
            pi_0, v_0 = self.neural_net.predictor(s[..., 0])

            # Collect predictions of the form: [w_i * 1 / K, v, r, pi] for each forward step k...K
            predictions = [(sample_weights, v_0, None, pi_0)]
            for t in range(actions.shape[1]):  # Shape (batch_size, K, action_size)
                r, s = self.neural_net.dynamics([s[..., 0], actions[:, t, :]])
                pi, v = self.neural_net.predictor(s[..., 0])

                predictions.append((tf.divide(sample_weights, len(actions)), v, r, pi))
                s = scale_gradient(s, 1 / 2)

            for t in range(len(predictions)):  # Length = 1 + K (root + hypothetical forward steps)
                gradient_scale, vs, rs, pis = predictions[t]
                t_vs, t_rs, t_pis = target_vs[t, ...], target_rs[t, ...], target_pis[t, ...]

                r_loss = scalar_loss(rs, t_rs) if t > 0 else tf.constant(0, dtype=tf.float32)
                v_loss = scalar_loss(vs, t_vs)
                pi_loss = scalar_loss(pis, t_pis)

                step_loss = r_loss + v_loss + pi_loss
                total_loss += tf.reduce_sum(scale_gradient(step_loss, gradient_scale))

            return total_loss
        return loss

    def train(self, examples: typing.List) -> float:
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples of the form (observation_trajectory,
                      action_trajectory, targets, loss_scale). Here targets is another
                      tuple comprised of the trajectories of (v, r, pi).
        """
        pass

    def get_variables(self) -> typing.List:
        """
        Yield a list of all trainable variables within the model

        Returns:
            variable_list: A list of all tf.Variables within the entire MuZero model.
        """
        pass

    def encode(self, observations: np.ndarray) -> np.ndarray:
        """
        Input:
            observations: A trajectory/ observation of an environment (in canonical form).

        Returns:
            s_0: A neural encoding of the environment.
        """
        pass

    def forward(self, latent_state: np.ndarray, action: int) -> typing.Tuple[float, np.ndarray]:
        """
        Input:
            latent_state: A neural encoding of the environment at step k: s_k.
            action: A (encoded) action to perform on the latent state

        Returns:
            r: The immediate predicted reward of the environment
            s_(k+1): A new 'latent_state' resulting from performing the 'action' in
                the latent_state.
        """
        pass

    def predict(self, latent_state: np.ndarray) -> typing.Tuple[np.ndarray, float]:
        """
        Input:
            latent_state: A neural encoding of the environment's.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float that gives the value of the current board
        """
        pass

    def save_checkpoint(self, folder: str, filename: str) -> None:
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder: str, filename: str) -> None:
        """
        Loads parameters of the neural network from folder/filename
        """
        pass
