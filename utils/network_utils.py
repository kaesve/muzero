"""

"""
import typing

from keras.layers import Layer
from keras import backend as k

import numpy as np


class MinMaxScaler(Layer):
    """
    Keras layer to MinMax scale an incoming tensor to the range [D, 1 + D].

    The transformation is defined as:
        s = (s - min(s)) / (max(s) - min(s) + e) + D
    here s is the tensor passed to the layer, e is a constant for numerical
    stability and D is a minimum value for the tensor.

    D is derived by the dimensions of s, D = prod(dim(s)) ** shrinkage.
    """

    def __init__(self, epsilon: float = 1.0, shrinkage: float = 1.0, safe: bool = False) -> None:
        """
        :param epsilon: float additive constant in the division operator
        :param shrinkage: float exponent for controlling the minimum value of the output tensor (higher = smaller s).
        """
        super().__init__()
        self.epsilon = epsilon
        self.shrinkage = shrinkage
        self.safe = safe

        self.D = float()
        self.shape = tuple()

    def build(self, input_shape: np.ndarray) -> None:
        """Initialize shapes and D within the computation graph"""
        self.D = 1.0 / (np.prod(input_shape[1:]) ** self.shrinkage)
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

        if self.safe:
            return (inputs - tensor_min) / (tensor_max - tensor_min + self.epsilon) + self.D

        return (inputs - tensor_min) / (tensor_max - tensor_min)
