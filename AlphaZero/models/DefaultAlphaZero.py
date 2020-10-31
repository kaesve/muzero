import sys
import typing

import numpy as np
import tensorflow as tf

from AlphaZero.AlphaNeuralNet import AlphaZeroNeuralNet
from utils.loss_utils import support_to_scalar, scalar_to_support
from utils import DotDict
from .architectures import *

sys.path.append('../../..')


models = {
    'Hex': BuildHexNet,
    'Othello': BuildOthelloNet,
    'Gym': BuildGymNet
}


class DefaultAlphaZero(AlphaZeroNeuralNet):

    def __init__(self, game, net_args: DotDict, architecture: str) -> None:
        """

        :param game:
        :param net_args:
        """
        super().__init__(game, net_args, models[architecture])
        self.action_size = game.getActionSize()
        self.architecture = architecture

    def train(self, examples: typing.List) -> None:
        """
        examples: list of examples, each example is of form (state, (pi, v), loss_scale)
        """
        observations, targets, loss_scale = list(zip(*examples))
        target_pis, target_vs = list(map(np.asarray, zip(*targets)))

        # ```np.asarray``` does not copy data contained within iterable
        observations = np.asarray(observations)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        priorities = np.asarray(loss_scale) * self.net_args.batch_size  # Undo 1/N normalization.

        # Cast to distribution
        target_vs = scalar_to_support(target_vs, self.net_args.support_size)

        total_loss, pi_loss, v_loss = self.neural_net.model.train_on_batch(
            x=observations, y=[target_pis, target_vs], sample_weight=[priorities, priorities])
        l2_norm = tf.reduce_sum([tf.nn.l2_loss(x) for x in self.neural_net.model.get_weights()])

        self.monitor.log(total_loss, "total loss")
        self.monitor.log(pi_loss, "pi_loss")
        self.monitor.log(v_loss, "v_loss")
        self.monitor.log(l2_norm, "l2_norm")

        self.steps += 1

    def predict(self, observation: np.ndarray) -> typing.Tuple[np.ndarray, float]:
        """
        board: np array with board
        """
        # preparing input
        observation = observation[np.newaxis, :, :]

        pi, v = self.neural_net.model.predict(observation)
        v_real = support_to_scalar(v, self.net_args.support_size)

        return pi[0], np.asscalar(v_real)
