"""

"""
import os
import numpy as np
import sys
import typing

from utils.loss_utils import support_to_scalar, scalar_to_support, cast_to_tensor
from MuZero.MuNeuralNet import MuZeroNeuralNet
from .architectures import *

from utils.storage import DotDict

sys.path.append('../../..')


models = {
    "Gym": BuildGymNet,
    "Hex": BuildHexNet,
    "Atari": BuildAtariNet
}


class MuZeroDefault(MuZeroNeuralNet):
    """

    """

    def __init__(self, game, net_args: DotDict, architecture: str) -> None:
        """

        :param game:
        :param net_args:
        """
        super().__init__(game, net_args, models[architecture])
        self.action_size = game.getActionSize()
        self.architecture = architecture

    def get_variables(self) -> typing.List:
        """ Get all trainable parameters defined by the neural network """
        parts = (self.neural_net.encoder, self.neural_net.predictor, self.neural_net.dynamics)
        return [v for v_list in map(lambda n: n.weights, parts) for v in v_list]

    def train(self, examples: typing.List) -> float:
        """
        Format the data contained in examples for computing the loss
        :param examples:
        :return:
        """
        # Unpack and transform data for loss computation.
        observations, actions, targets, sample_weight = list(zip(*examples))

        actions, sample_weight = np.array(actions), np.array(sample_weight)

        # Unpack and encode targets. All target shapes are of the form [time, batch_size, categories]
        target_vs, target_rs, target_pis = list(map(np.array, zip(*targets)))

        target_vs = np.array([scalar_to_support(target_vs[:, t], self.net_args.support_size)
                              for t in range(target_vs.shape[-1])])
        target_rs = np.array([scalar_to_support(target_rs[:, t], self.net_args.support_size)
                              for t in range(target_rs.shape[-1])])
        target_pis = np.swapaxes(target_pis, 0, 1)

        # Pack formatted inputs as tensors.
        data = [cast_to_tensor(x) for x in [observations, actions, target_vs, target_rs, target_pis, sample_weight]]

        # Get the tf computation graph for the loss given the data.
        loss = self.loss_function(*data)

        # Perform an optimization step.
        _ = self.optimizer.minimize(loss, self.get_variables, name=f'MuZeroDefault_{self.architecture}')
        self.steps += 1

    def initial_inference(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float]:
        observations = observations[np.newaxis, ...]
        s_0, pi, v = self.neural_net.forward.predict_on_batch(observations)

        v_real = support_to_scalar(v, self.net_args.support_size)

        return s_0[0], pi[0], np.ndarray.item(v_real)

    def recurrent_inference(self, latent_state: np.ndarray, action: int) -> typing.Tuple[float, np.ndarray,
                                                                                         np.ndarray, float]:
        a_plane = np.zeros(self.action_size)
        a_plane[action] = 1

        latent_state = latent_state[np.newaxis, ...]
        a_plane = a_plane[np.newaxis, ...]

        r, s_next, pi, v = self.neural_net.recurrent.predict_on_batch([latent_state, a_plane])

        v_real = support_to_scalar(v, self.net_args.support_size)
        r_real = support_to_scalar(r, self.net_args.support_size)

        return np.ndarray.item(r_real), s_next[0], pi[0], np.ndarray.item(v_real)
