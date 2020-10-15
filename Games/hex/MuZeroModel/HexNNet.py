"""
In this python file we modified the class to enable generation of multiple
neural architectures by modifying the argument dictionary.

For the details of the neural architectures, we refer to our report.
:see: main_experimenter.py
"""

import sys

from keras.models import *
from keras.layers import *

from utils.network_utils import MinMaxScaler

sys.path.append('../..')


class HexNNet:

    def __init__(self, game, args):
        # Network arguments
        self.board_x, self.board_y, self.planes = game.getDimensions(game.Observation.HEURISTIC)
        self.action_size = game.getActionSize()
        self.args = args

        # s: batch_size x time x state_x x state_y
        self.observation_history = Input(shape=(self.board_x, self.board_y, self.planes * self.args.observation_length))
        # a: one hot encoded vector of shape batch_size x (state_x * state_y)
        self.action_plane = Input(shape=(self.action_size,))
        # s': batch_size  x board_x x board_y x 1
        self.latent_state = Input(shape=(self.board_x, self.board_y))

        action_plane = Lambda(lambda x: x[..., :-1], output_shape=(self.board_x * self.board_y,),
                              input_shape=(self.action_size,))(self.action_plane)  # Omit resignation
        action_plane = Reshape((self.board_x, self.board_y, 1))(action_plane)
        latent_state = Reshape((self.board_x, self.board_y, 1))(self.latent_state)

        self.s = self.encoder(self.observation_history)
        self.r, self.s_next = self.dynamics(latent_state, action_plane)

        self.pi, self.v = self.predictor(latent_state)

        self.encoder = Model(inputs=self.observation_history, outputs=self.s)
        self.dynamics = Model(inputs=[self.latent_state, self.action_plane], outputs=[self.r, self.s_next])
        self.predictor = Model(inputs=self.latent_state, outputs=[self.pi, self.v])

    def build_model(self, tensor_in):
        def conv_block(n, x):  # Recursively builds a convolutional tower of height n.
            if n > 0:
                return conv_block(n - 1, Activation('relu')(BatchNormalization()(Conv2D(
                    self.args.num_channels, 3, padding='same', use_bias=False)(x))))
            return x

        def dense_sequence(n, x):  # Recursively builds a Fully Connected sequence of length n.
            if n > 0:
                return dense_sequence(n - 1, Dropout(self.args.dropout)(Activation('relu')(
                    Dense(self.args.size_dense)(x))))
            return x

        conv_block = conv_block(self.args.num_towers, tensor_in)
        flattened = Flatten()(conv_block)
        fc_sequence = dense_sequence(self.args.len_dense, flattened)
        return fc_sequence

    def encoder(self, observations):
        out_tensor = self.build_model(observations)

        s_fc_latent = Dense(self.board_x * self.board_y, activation='linear', name='s_0')(out_tensor)
        latent_state = Reshape((self.board_x, self.board_y, 1))(s_fc_latent)
        latent_state = MinMaxScaler()(latent_state)

        return latent_state  # 2-dimensional 1-time step latent state. (Encodes history of images into one state).

    def dynamics(self, encoded_state, action_plane):
        stacked = Concatenate(axis=-1)([encoded_state, action_plane])
        reshaped = Reshape((self.board_x, self.board_y, -1))(stacked)
        out_tensor = self.build_model(reshaped)

        s_fc_latent = Dense(self.board_x * self.board_y, activation='linear', name='s_next')(out_tensor)
        latent_state = Reshape((self.board_x, self.board_y, 1))(s_fc_latent)
        latent_state = MinMaxScaler()(latent_state)

        r = Dense(1, activation='linear', name='r')(out_tensor) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='r')(out_tensor)

        return r, latent_state

    def predictor(self, latent_state):
        out_tensor = self.build_model(latent_state)

        pi = Dense(self.action_size, activation='softmax', name='pi')(out_tensor)
        v = Dense(1, activation='tanh', name='v')(out_tensor) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(out_tensor)

        return pi, v
