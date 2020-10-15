"""

"""
import typing

from keras.layers import Layer
from keras import backend as k

import numpy as np


class MinMaxScaler(Layer):
    """
    Keras layer to MinMax scale an incoming tensor to the range [0, 1].

    The transformation is defined as:
        s = (s - min(s)) / (max(s) - min(s) + e)
    here s is the tensor passed to the layer, e is a small constant for numerical stability.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        """
        :param epsilon: float additive constant in the division operator
        """
        super().__init__()
        self.epsilon = epsilon
        self.shape = tuple()

    def build(self, input_shape: np.ndarray) -> None:
        """Initialize shapes within the computation graph"""
        self.shape = input_shape

    def call(self, inputs, **kwargs) -> typing.Callable:
        """
        MinMax normalize the given inputs with minimum value 1 / dimensions.
        Normalization is performed strictly over one example (not a batch).
        :param inputs:
        :param kwargs:
        :return:
        """
        tensor_min = k.min(inputs, axis=np.arange(1, len(self.shape)), keepdims=True)
        tensor_max = k.max(inputs, axis=np.arange(1, len(self.shape)), keepdims=True)

        return (inputs - tensor_min) / (tensor_max - tensor_min + 1e-8)
