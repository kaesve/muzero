"""
Defines an implementation of MuZeroNeuralNet that overrides the DefaultMuZero implementation.
This class handles training of a latent-to-real-state decoder jointly with the MuZero model.

To make use of this class, ensure your neural network class has a 'decoder' attribute that defines
a keras Model to a trainable neural network decoder. Then simply specify whether or not to train a decoder
in the algorithm parameter .json file.

Notes:
 -  Base implementation done 15/11/2020
 -  Documentation 16/11/2020
"""
import typing

import tensorflow as tf

from utils import DotDict
from .DefaultMuZero import DefaultMuZero
from utils.loss_utils import scale_gradient, safe_l2norm, scalar_loss


class DecoderMuZero(DefaultMuZero):
    """
    Class to override the loss functions for MuZero to jointly train a latent-to-real-state neural decoder.
    Based on the provided scaling factor, 'dynamics_penalty', the user governs how much gradient is propagated
    from the decoder to the rest of the MuZero neural network. A 'dynamics_penalty' of zero will still train
    the decoder, but not propagate any gradient from the decoder to the neural network. A 'dynamics_penalty' of one
    will make the training of the dynamics model similar to a recurrent sequence auto-encoder.
    """

    def __init__(self, game, net_args: DotDict, architecture: str) -> None:
        super().__init__(game, net_args, architecture)

        # Ensure that the provided neural network has an available decoder.
        if not hasattr(self.neural_net, 'decoder'):
            raise NotImplementedError(f"Provided neural network {architecture} has no defined 'decoder' attribute. "
                                      f"Either set the 'latent_decoder' option to 'false' in the .json, or implement "
                                      f"a decoding neural network model in your network constructor.")

    def get_variables(self) -> typing.List:
        """ Get all trainable parameters defined by the neural network + decoder weights """
        parts = (self.neural_net.encoder, self.neural_net.predictor, self.neural_net.dynamics, self.neural_net.decoder)
        return [v for v_list in map(lambda n: n.weights, parts) for v in v_list]

    @tf.function
    def unroll(self, observations: tf.Tensor, actions: tf.Tensor) -> \
            typing.List[typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        Overrides super function to additionally predict state-observations using a decoder model.

        The magnitude of gradient that passes through the decoder model back to the MuZero dynamics model
        is governed by the dynamics penalty.

        :param observations: tf.Tensor in R^(batch_size x width x height x (depth * time))
        :param actions: tf.Tensor consisting of one-hot-encoded actions in {0, 1}^(batch_size x K x |action_space|)
        :return: List of tuples containing the hidden state, value predictions and loss-scale for each unrolled step.
        """
        # Root inference. Collect predictions of the form: [w_i / K, o_k, v, r, pi] for each forward step k = 0...K
        s, pi_0, v_0 = self.neural_net.forward(observations)

        # Decouple latent state from default unrolling graph to accordingly distribute (scaled) gradients.
        s_decoupled = scale_gradient(s, self.net_args.dynamics_penalty)
        o_t = self.neural_net.decoder(s_decoupled)

        # Note: Root can be a terminal state. Loss scale for the root head is 1.0 instead of 1 / K.
        predictions = [(1.0, o_t, v_0, 0, pi_0)]
        for k in range(actions.shape[1]):
            r, s, pi, v = self.neural_net.recurrent([s, actions[:, k, :]])

            # Decouple latent state from default unrolling graph to accordingly distribute (scaled) gradients.
            s_decoupled = scale_gradient(s, self.net_args.dynamics_penalty)
            o_k = self.neural_net.decoder(s_decoupled)

            predictions.append((1.0 / actions.shape[1], o_k, v, r, pi))

            # Scale the gradient at the start of the dynamics function by 1/2
            s = scale_gradient(s, 0.5)

        return predictions

    @tf.function
    def loss_function(self, observations, actions, target_vs, target_rs, target_pis, target_observations,
                      sample_weights) -> typing.Tuple[tf.Tensor, typing.List]:
        """
        Overrides super function to compute the loss for decoding the unrolled latent-states back to true future
        observations.

        :param observations: tf.Tensor in R^(batch_size x width x height x (depth * time)). Stacked state observations.
        :param actions: tf.Tensor in {0, 1}^(batch_size x K x |action_space|). One-hot encoded actions for unrolling.
        :param target_vs: tf.Tensor either in [0,1] or R with dimensions (K x batch_size x support_size)
        :param target_rs: tf.Tensor either in [0,1] or R with dimensions (K x batch_size x support_size)
        :param target_pis: tf.Tensor either in [0,1] or R with dimensions (K x batch_size x |action_space|)
        :param target_observations: tf.Tensor of same dimensions of observations for each unroll step in axis 1.
        :param sample_weights: tf.Tensor in [0, 1]^(batch_size). Of the form (batch_size * priority) ^ (-beta)
        :return: tuple of a tf.Tensor and a list of tf.Tensors containing the total loss and piecewise losses.
        :see: MuNeuralNet.unroll
        """
        loss_monitor = []  # Collect losses for logging.

        # Sum over target probabilities. Absorbing states should have a zero sum --> leaf node.
        absorb_k = 1.0 - tf.reduce_sum(target_pis, axis=-1)

        # Root inference. Collect predictions of the form: [w_i / K, s, v, r, pi] for each forward step k = 0...K
        predictions = self.unroll(observations, actions)

        # Perform loss computation for each unrolling step.
        total_loss = tf.constant(0.0, dtype=tf.float32)
        for k in range(len(predictions)):  # Length = 1 + K (root + hypothetical forward steps)
            loss_scale, p_obs, vs, rs, pis = predictions[k]
            t_vs, t_rs, t_pis = target_vs[k, ...], target_rs[k, ...], target_pis[k, ...]
            absorb = absorb_k[k, :]

            # Decoder target observations
            t_obs = observations if k == 0 else target_observations[:, (k - 1), ...]

            # Calculate losses per head. Cancel gradients in prior for absorbing states, keep gradients for r and v.
            r_loss = scalar_loss(rs, t_rs) if (k > 0 and self.fit_rewards) else tf.constant(0, dtype=tf.float32)
            v_loss = scalar_loss(vs, t_vs)
            pi_loss = scalar_loss(pis, t_pis) * (1.0 - absorb)
            o_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(t_obs, p_obs), axis=(1, 2))

            step_loss = scale_gradient(r_loss + v_loss + pi_loss + o_loss, loss_scale * sample_weights)
            total_loss += tf.reduce_sum(step_loss)  # Actually averages over batch : see sample_weights.

            # Logging
            loss_monitor.append((v_loss, r_loss, pi_loss, absorb, o_loss))

        # Penalize magnitude of weights using l2 norm
        l2_norm = tf.reduce_sum([safe_l2norm(x) for x in self.get_variables()])
        total_loss += self.net_args.l2 * l2_norm

        return total_loss, loss_monitor
