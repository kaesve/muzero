"""

"""
import typing

import numpy as np
from keras.layers import Layer, LeakyReLU, Activation, Dropout, Conv2D, Dense, Flatten, Lambda
from keras import backend as k


class MinMaxScaler(Layer):
    """
    Keras layer to MinMax scale an incoming tensor to the range [0, 1].

    The transformation is defined as:
        s = (s - min(s)) / (max(s) - min(s) + e)
    here s is the tensor passed to the layer, e is a small constant for numerical stability.
    """

    def __init__(self, epsilon: float = 1e-5) -> None:
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

        return (inputs - tensor_min) / (tensor_max - tensor_min + self.epsilon)


class Crafter:

    def __init__(self, args):
        self.args = args
        self.activation = lambda: (Activation(args.activation)
                                   if args.activation != "leakyrelu" else LeakyReLU(alpha=0.2))

    def conv_residual_tower(self, n: int, x, left_n: int = 2, right_n: int = 0):
        assert left_n > 0, "Residual network must have at least a conv block larger than 0."

        if n > 0:
            left = self.conv_tower(left_n - 1, x)
            if left_n - 1 > 0:
                left = (Conv2D(self.args.num_channels, 3, padding='same', use_bias=False)(left))

            right = self.conv_tower(right_n - 1, x)
            if right_n - 1 > 0:
                right = (Conv2D(self.args.num_channels, 3, padding='same', use_bias=False)(right))

            # TODO: Create GitHub Issue: Add layer produces NameError in tf graph. Equivalent Lambda K.sum does work.
            merged = Lambda(lambda var: k.sum(var, axis=0))([left, right])
            out_tensor = self.activation()(merged)

            return self.conv_residual_tower(n - 1, out_tensor, left_n, right_n)

        return x

    def conv_tower(self, n: int, x):
        """ Recursively builds a convolutional tower of height n. """
        if n > 0:
            return self.conv_tower(n - 1, self.activation()((Conv2D(
                self.args.num_channels, 3, padding='same', use_bias=False)(x))))
        return x

    def dense_sequence(self, n: int, x):
        """ Recursively builds a Fully Connected sequence of length n. """
        if n > 0:
            return self.dense_sequence(n - 1, Dropout(self.args.dropout)(self.activation()(
                Dense(self.args.size_dense)(x))))
        return x

    def build_conv_block(self, tensor_in):
        conv_block = self.conv_tower(self.args.num_towers, tensor_in)
        flattened = Flatten()(conv_block)
        fc_sequence = self.dense_sequence(self.args.num_dense, flattened)
        return fc_sequence
