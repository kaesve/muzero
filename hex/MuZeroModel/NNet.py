"""

"""
import os
import numpy as np
import sys
import typing

from utils.loss_utils import support_to_scalar, scalar_to_support, cast_to_tensor
from MuZero.MuNeuralNet import MuZeroNeuralNet
from .HexNNet import HexNNet as NetBuilder

from Game import Game
from utils.storage import DotDict

sys.path.append('../..')


class NNetWrapper(MuZeroNeuralNet):
    """

    """

    def __init__(self, game, net_args: DotDict) -> None:
        """

        :param game:
        :param net_args:
        """
        super().__init__(game, net_args, NetBuilder)
        self.board_x, self.board_y = game.getDimensions()
        self.action_size = game.getActionSize()

    def get_variables(self) -> typing.List:
        """

        :return:
        """
        parts = (self.neural_net.encoder, self.neural_net.predictor, self.neural_net.dynamics)
        return [v for v_list in map(lambda n: n.weights, parts) for v in v_list]

    def train(self, examples: typing.List) -> float:
        """

        :param examples:
        :return:
        """

        def encode_returns(x: np.ndarray) -> np.ndarray:
            """"""
            return scalar_to_support(x, self.net_args.support_size)

        # Unpack and transform data for loss computation.
        observations, actions, targets, sample_weight = list(zip(*examples))
        actions, sample_weight = np.array(actions), np.array(sample_weight)

        # Unpack and encode targets. All target shapes are of the form [time, batch_size, categories]
        target_vs, target_rs, target_pis = list(map(np.array, zip(*targets)))
        target_vs = np.array([encode_returns(target_vs[:, t]) for t in range(target_vs.shape[-1])])
        target_rs = np.array([encode_returns(target_rs[:, t]) for t in range(target_rs.shape[-1])])
        target_pis = np.swapaxes(target_pis, 0, 1)

        # Pack formatted inputs as tensors.
        data = [cast_to_tensor(x) for x in [observations, actions, target_vs, target_rs, target_pis, sample_weight]]

        loss = self.loss_function(*data)

        # Perform an optimization step.
        _ = self.optimizer.minimize(loss, self.get_variables)
        return loss()  # Returns loss contained within a tf.tensor

    def encode(self, observations: np.ndarray) -> np.ndarray:
        """

        :param observations:
        :return:
        """
        observations = observations[np.newaxis, ...]
        latent_state = self.neural_net.encoder.predict(observations)[0]
        return latent_state

    def forward(self, latent_state: np.ndarray, action: int) -> typing.Tuple[float, np.ndarray]:
        """

        :param latent_state:
        :param action:
        :return:
        """
        a_plane = np.zeros(self.action_size)
        a_plane[action] = 1

        latent_state = latent_state.reshape((-1, self.board_x, self.board_y))
        a_plane = a_plane[np.newaxis, ...]

        r, s_next = self.neural_net.dynamics.predict([latent_state, a_plane])

        r_real = support_to_scalar(r, self.net_args.support_size)

        return np.ndarray.item(r_real), s_next[0]

    def predict(self, latent_state: np.ndarray) -> typing.Tuple[np.ndarray, float]:
        """

        :param latent_state:
        :return:
        """
        latent_state = latent_state.reshape((-1, self.board_x, self.board_y))
        pi, v = self.neural_net.predictor.predict(latent_state)

        v_real = support_to_scalar(v, self.net_args.support_size)

        return pi[0], np.ndarray.item(v_real)

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """

        :param folder:
        :param filename:
        :return:
        """
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

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """

        :param folder:
        :param filename:
        :return:
        """
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
