"""

"""
import typing

import numpy as np
from keras.layers import Layer, Activation, BatchNormalization, Conv2D, Dropout, Dense, Flatten
from keras import backend as k


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


class Crafter:

    def __init__(self, args):
        self.args = args

    def conv_tower(self, n: int, x):
        """ Recursively builds a convolutional tower of height n. """
        if n > 0:
            return self.conv_tower(n - 1, Activation('relu')(BatchNormalization()(Conv2D(
                self.args.num_channels, 3, padding='same', use_bias=False)(x))))
        return x

    def dense_sequence(self, n: int, x):
        """ Recursively builds a Fully Connected sequence of length n. """
        if n > 0:
            return self.dense_sequence(n - 1, Dropout(self.args.dropout)(Activation(self.args.dense_activation)(
                Dense(self.args.size_dense)(x))))
        return x

    def build_conv_block(self, tensor_in):
        conv_block = self.conv_tower(self.args.num_towers, tensor_in)
        flattened = Flatten()(conv_block)
        fc_sequence = self.dense_sequence(self.args.num_dense, flattened)
        return fc_sequence
