"""
In this python file we modified the class to enable generation of multiple
neural architectures by modifying the argument dictionary.

For the details of the neural architectures, we refer to our report.
:see: main_experimenter.py
"""

import sys

from keras.models import Model
from keras.layers import *

from utils.network_utils import MinMaxScaler, Crafter

sys.path.append('../..')


class HexNNet:

    def __init__(self, game, args):
        # Network arguments
        self.board_x, self.board_y, self.planes = game.getDimensions(game.Representation.HEURISTIC)
        self.action_size = game.getActionSize()
        self.args = args
        self.crafter = Crafter(args)

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

        self.s = self.build_encoder(self.observation_history)
        self.r, self.s_next = self.build_dynamics(latent_state, action_plane)
        self.pi, self.v = self.build_predictor(latent_state)

        self.encoder = Model(inputs=self.observation_history, outputs=self.s)
        self.dynamics = Model(inputs=[self.latent_state, self.action_plane], outputs=[self.r, self.s_next])
        self.predictor = Model(inputs=self.latent_state, outputs=[self.pi, self.v])

        self.forward = Model(inputs=self.observation_history, outputs=[self.s, *self.predictor(self.s)])
        self.recurrent = Model(inputs=[self.latent_state, self.action_plane],
                               outputs=[self.r, self.s_next, *self.predictor(self.s_next)])

    def build_encoder(self, observations):
        out_tensor = self.crafter.conv_tower(self.args.num_towers, observations)

        latent_state = Activation('relu')(Conv2D(1, 3, padding='same', use_bias=False)(out_tensor))
        latent_state = MinMaxScaler()(latent_state)

        return latent_state  # 2-dimensional 1-time step latent state. (Encodes history of images into one state).

    def build_dynamics(self, encoded_state, action_plane):
        stacked = Concatenate(axis=-1)([encoded_state, action_plane])
        reshaped = Reshape((self.board_x, self.board_y, -1))(stacked)
        out_tensor = self.crafter.conv_tower(self.args.num_towers, reshaped)

        latent_state = Activation('relu')(Conv2D(1, 3, padding='same', use_bias=False)(out_tensor))
        latent_state = MinMaxScaler()(latent_state)

        flat = Flatten()(out_tensor)
        r = Dense(1, activation='linear', name='r')(flat) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='r')(flat)

        return r, latent_state

    def build_predictor(self, latent_state):
        out_tensor = self.crafter.conv_tower(self.args.num_towers, latent_state)

        flat = Flatten()(out_tensor)

        pi = Dense(self.action_size, activation='softmax', name='pi')(flat)
        v = Dense(1, activation='tanh', name='v')(flat) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(flat)

        return pi, v
