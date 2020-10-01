import os
import numpy as np
import sys

from utils.loss_utils import support_to_scalar
from MuZero.MuNeuralNet import MuZeroNeuralNet
from .HexNNet import HexNNet as NetBuilder

sys.path.append('../..')


import tensorflow as tf



def scale_gradient(tensor, scale):
  """Scales the gradient for the backward pass."""
  return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def scalar_loss(prediction, target):
    #TODO
    pass

class NNetWrapper(MuZeroNeuralNet):
    def __init__(self, game, net_args):
        super().__init__(game)
        self.net_args = net_args
        self.neural_net = NetBuilder(game, net_args)
        self.board_x, self.board_y = game.getDimensions()
        self.action_size = game.getActionSize()

        
        self.optimizer = tf.train.MomentumOptimizer(net_args.lr, net_args.momentum)

    def train(self, examples):
        """
        """

        total_loss = 0

        for observations, actions, targets in examples:
            latent_state = self.encode(observations)
            value, policy_logits = self.predict(latent_state)

            predictions = [(1, value, 0, policy_logits)]

            # build predictions for k future steps
            for action in actions:
                reward, latent_state = self.forward(latent_state, action)
                value, policy_logits = self.predict(latent_state)

                predictions.append((1/len(actions), value, reward, policy_logits))

                latent_state = scale_gradient(latent_state)
            
            for prediction, target in zip(predictions, targets):
                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                step_loss = (
                    scalar_loss(value, target_value) +
                    scalar_loss(reward, target_reward) +
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=policy_logits, labels=target_policy))

                total_loss += scale_gradient(step_loss, gradient_scale)

            for weights in self.neural_net.get_weights():
                total_loss += weight_decay * tf.nn.l2_loss(weights)

        self.optimizer.minimize(total_loss)


    def encode(self, observations):  # TODO: Scaling of hidden activations
        observations = observations[np.newaxis, ...]
        return self.neural_net.encoder.predict(observations)[0]

    def forward(self, latent_state, action):  # TODO: Scaling of hidden activations
        a_plane = np.zeros((self.board_x, self.board_y))
        a_plane[action // self.board_x][action % self.board_y] = 1

        latent_state = latent_state.reshape((-1, self.board_x, self.board_y))
        a_plane = a_plane.reshape((-1, self.board_x, self.board_y))

        r, s_next = self.neural_net.dynamics.predict([latent_state, a_plane])

        r_real = support_to_scalar(r[0], self.net_args.support_size)

        return r_real, s_next[0]

    def predict(self, latent_state):
        """
        board: np array with board
        """
        latent_state = latent_state.reshape((-1, self.board_x, self.board_y))
        pi, v = self.neural_net.predictor.predict(latent_state)

        v_real = support_to_scalar(v[0], self.net_args.support_size)

        return pi[0], v_real

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        representation_path = os.path.join(folder, 'r_'+filename)
        dynamics_path = os.path.join(folder, 'd_'+filename)
        predictor_path = os.path.join(folder, 'p_'+filename)
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
