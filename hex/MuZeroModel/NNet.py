import os
import numpy as np
import sys

from utils.loss_utils import support_to_scalar, scalar_to_support, scalar_loss
from MuZero.MuNeuralNet import MuZeroNeuralNet
from .HexNNet import HexNNet as NetBuilder


import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

sys.path.append('../..')


class NNetWrapper(MuZeroNeuralNet):
    def __init__(self, game, net_args):
        super().__init__(game)
        self.net_args = net_args
        self.neural_net = NetBuilder(game, net_args)
        self.board_x, self.board_y = game.getDimensions()
        self.action_size = game.getActionSize()

        # self.optimizer = tf.train.MomentumOptimizer(net_args.lr, net_args.momentum)
        self.optimizer = tf.optimizers.Adam(net_args.lr)

    def get_variables(self):
        parts = (self.neural_net.encoder, self.neural_net.predictor, self.neural_net.dynamics)
        return [v for v_list in map(lambda n: n.weights, parts) for v in v_list]

    def train(self, examples):
        """
        """
        def encode(x):
            return scalar_to_support(x, self.net_args.support_size)

        def scale_gradient(tensor, scale):
            """Scales the gradient for the backward pass."""
            return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

        @tf.function
        def loss():
            total_loss = tf.constant(0, dtype=tf.float32)

            # Root inference
            s = self.neural_net.encoder(observations)
            pi_0, v_0 = self.neural_net.predictor(s[..., 0])

            # Collect predictions of the form: [w_i * 1 / K, v, r, pi] for each forward step k...K
            predictions = [(sample_weight, v_0, None, pi_0)]
            for t in range(actions.shape[1]):  # Shape (batch_size, K, action_size)
                r, s = self.neural_net.dynamics([s[..., 0], actions[:, t, :]])
                pi, v = self.neural_net.predictor(s[..., 0])

                predictions.append((sample_weight / len(actions), v, r, pi))
                s = scale_gradient(s, 1 / 2)

            for t in range(len(predictions)):  # Length = 1 + K (root + hypothetical forward steps)
                gradient_scale, vs, rs, pis = predictions[t]
                t_vs, t_rs, t_pis = target_vs[t, ...], target_rs[t, ...], target_pis[t, ...]

                r_loss = scalar_loss(rs, t_rs) if t > 0 else 0
                v_loss = scalar_loss(vs, t_vs)
                pi_loss = scalar_loss(pis, t_pis)

                step_loss = r_loss + v_loss + pi_loss
                total_loss += tf.reduce_sum(scale_gradient(step_loss, gradient_scale))

            return total_loss

        # Unpack and transform data for loss computation.
        observations, actions, targets, sample_weight = list(zip(*examples))
        actions, sample_weight = np.array(actions), np.array(sample_weight)

        # Unpack and encode targets. All target shapes are of the form [time, batch_size, categories]
        target_vs, target_rs, target_pis = list(map(np.array, zip(*targets)))

        target_vs = np.array([encode(target_vs[:, t]) for t in range(target_vs.shape[-1])])
        target_rs = np.array([encode(target_rs[:, t]) for t in range(target_rs.shape[-1])])
        target_pis = np.swapaxes(target_pis, 0, 1)

        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        sample_weight = tf.convert_to_tensor(sample_weight, dtype=tf.float32)
        target_rs = tf.convert_to_tensor(target_rs, dtype=tf.float32)
        target_vs = tf.convert_to_tensor(target_vs, dtype=tf.float32)
        target_pis = tf.convert_to_tensor(target_pis, dtype=tf.float32)

        l = loss()
        print("before opt", l)

        self.optimizer.minimize(loss, self.get_variables)

        print("after opt", loss())

    def encode(self, observations):
        observations = observations[np.newaxis, ...]
        latent_state = self.neural_net.encoder.predict(observations)[0]
        return latent_state

    def forward(self, latent_state, action):
        a_plane = np.zeros(self.action_size)
        a_plane[action] = 1

        latent_state = latent_state.reshape((-1, self.board_x, self.board_y))
        a_plane = a_plane[np.newaxis, ...]

        r, s_next = self.neural_net.dynamics.predict([latent_state, a_plane])

        r_real = support_to_scalar(r, self.net_args.support_size)

        return np.asscalar(r_real), s_next[0]

    def predict(self, latent_state):
        """
        board: np array with board
        """
        latent_state = latent_state.reshape((-1, self.board_x, self.board_y))
        pi, v = self.neural_net.predictor.predict(latent_state)

        v_real = support_to_scalar(v, self.net_args.support_size)

        return pi[0], np.asscalar(v_real)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        representation_path = os.path.join(folder, 'r_' + filename)
        dynamics_path = os.path.join(folder, 'd_' + filename)
        predictor_path = os.path.join(folder, 'p_' + filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.neural_net.encoder.save_weights(representation_path)
        self.neural_net.dynamics.save_weights(dynamics_path)
        self.neural_net.predictor.save_weights(predictor_path)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        representation_path = os.path.join(folder, 'r_' + filename)
        dynamics_path = os.path.join(folder, 'd_' + filename)
        predictor_path = os.path.join(folder, 'p_' + filename)

        if not os.path.exists(representation_path):
            raise FileNotFoundError("No AlphaZeroModel in path {}".format(representation_path))
        if not os.path.exists(dynamics_path):
            raise FileNotFoundError("No AlphaZeroModel in path {}".format(dynamics_path))
        if not os.path.exists(predictor_path):
            raise FileNotFoundError("No AlphaZeroModel in path {}".format(predictor_path))

        self.neural_net.encoder.load_weights(representation_path)
        self.neural_net.dynamics.load_weights(dynamics_path)
        self.neural_net.predictor.load_weights(predictor_path)
