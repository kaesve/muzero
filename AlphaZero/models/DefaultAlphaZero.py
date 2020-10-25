import sys
import typing

import numpy as np

from AlphaZero.AlphaNeuralNet import AlphaZeroNeuralNet
from utils.loss_utils import support_to_scalar, scalar_to_support
from utils.network_utils import CustomTensorBoard
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
        self.callbacks = [CustomTensorBoard(self)]

    def train(self, examples: typing.List, steps: int) -> None:
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        observations, target_pis, target_vs = list(zip(*examples))

        # ```np.asarray``` does not copy data contained within iterable
        observations = np.asarray(observations)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        # Cast to distribution
        target_vs = scalar_to_support(target_vs, self.net_args.support_size)

        self.neural_net.model.fit(
            x=observations, y=[target_pis, target_vs],
            batch_size=self.net_args.batch_size, epochs=steps,
            verbose=0, callbacks=self.callbacks
        )

        self.steps += steps

    def predict(self, observation: np.ndarray) -> typing.Tuple[np.ndarray, float]:
        """
        board: np array with board
        """
        # preparing input
        observation = observation[np.newaxis, :, :]

        pi, v = self.neural_net.model.predict(observation)
        v_real = support_to_scalar(v, self.net_args.support_size)

        return pi[0], np.asscalar(v_real)
