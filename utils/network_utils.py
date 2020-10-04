from keras.layers import Layer
from keras import backend as k

import numpy as np


class MinMaxScaler(Layer):

    def __init__(self):
        super().__init__()
        self.epsilon = 1.0

        self.D = float()
        self.shape = tuple()

    def build(self, input_shape):
        self.D = 1.0 / np.prod(input_shape[1:])
        self.shape = input_shape

    def call(self, inputs, **kwargs):
        """
        MinMax normalize the given inputs with minimum value 1 / dimensions.
        Normalization is performed strictly over one example (not a batch).
        :param inputs:
        :param kwargs:
        :return:
        """
        tensor_min = k.min(inputs, axis=np.arange(1, len(self.shape)), keepdims=True)
        tensor_max = k.max(inputs, axis=np.arange(1, len(self.shape)), keepdims=True)

        return (inputs - tensor_min) / (tensor_max - tensor_min + self.epsilon) + self.D
