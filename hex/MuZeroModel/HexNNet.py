"""
In this python file we modified the class to enable generation of multiple
neural architectures by modifying the argument dictionary.

For the details of the neural architectures, we refer to our report.
:see: main_experimenter.py
"""

import sys

from keras.models import *
from keras.layers import *

sys.path.append('..')


class HexNNet:

    def __init__(self, game, args):
        # Network arguments
        self.board_x, self.board_y = game.getDimensions()
        self.action_size = game.getActionSize()
        self.args = args

        # s: batch_size x state_x x state_y x time
        self.observation_history = Input(shape=(self.board_x, self.board_y, self.args.observation_length))
        # a: batch_size x state_x x state_y x 1
        self.action_plane = Input(shape=(self.board_x, self.board_y))
        # s': batch_size  x board_x x board_y x 1
        self.latent_state = Input(shape=(self.board_x, self.board_y))

        # TODO: Check functionality during training
        action_plane = Reshape((self.board_x, self.board_y, 1))(self.action_plane)
        latent_state = Reshape((self.board_x, self.board_y, 1))(self.latent_state)

        self.s = self.encoder(self.observation_history)
        self.r, self.s_next = self.dynamics(latent_state, action_plane)

        self.pi, self.v = self.predictor(latent_state)

        self.encoder = Model(inputs=self.observation_history, outputs=self.s)
        self.dynamics = Model(inputs=[self.latent_state, self.action_plane], outputs=[self.r, self.s_next])
        self.predictor = Model(inputs=self.latent_state, outputs=[self.pi, self.v])

    def block(self, n, x):  # Recursively build a convolutional tower of height n.
        if n > 0:
            return self.block(n - 1, Activation('relu')(BatchNormalization()(Conv2D(
                self.args.num_channels, 3, padding='same', use_bias=False)(x))))
        return x

    def encoder(self, observations):
        h_conv2 = self.block(self.args.num_towers, observations)
        latent_state = Activation('relu')(BatchNormalization()(Conv2D(1, 3, padding='same', use_bias=False)(h_conv2)))
        return latent_state  # 2-dimensional 1-time step latent state. (Encodes history of images into one state).

    def dynamics(self, encoded_state, action_plane):
        stacked = Concatenate(axis=-1)([encoded_state, action_plane])
        reshaped = Reshape((self.board_x, self.board_y, -1))(stacked)
        h_conv2 = self.block(self.args.num_towers, reshaped)
        latent_state = Activation('relu')(BatchNormalization()(Conv2D(1, 3, padding='same', use_bias=False, name='s_next')(h_conv2)))

        flattened = Flatten()(latent_state)
        reward = Dense(self.args.support_size, activation='softmax', name='r')(flattened)

        return reward, latent_state

    def predictor(self, latent_state):
        h_conv2 = self.block(self.args.num_towers, latent_state)
        h_conv2_flat = Flatten()(h_conv2)
        s_fc1 = Dropout(self.args.dropout)(Activation('relu')(Dense(256)(h_conv2_flat)))

        pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc1)
        v = Dense(1, activation='tanh', name='v')(s_fc1)

        return pi, v
