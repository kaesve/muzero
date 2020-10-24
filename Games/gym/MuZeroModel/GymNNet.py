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


class GymNNet:

    def __init__(self, game, args):
        # Network arguments
        self.x, self.y, self.planes = game.getDimensions()
        self.latents = 8
        self.action_size = game.getActionSize()
        self.args = args

        # s: batch_size x time x state_x x state_y
        self.observation_history = Input(shape=(self.x, self.y, self.planes))
        # a: one hot encoded vector of shape batch_size x (state_x * state_y)
        self.action_tensor = Input(shape=(self.action_size, ))
        # s': batch_size  x board_x x board_y x 1
        self.latent_state = Input(shape=(self.latents, 1))

        observations = Reshape((self.x * self.y * self.planes, ))(self.observation_history)
        latent_state = Reshape((self.latents, ))(self.latent_state)

        # Build tensorflow computation graph
        self.s = self.build_encoder(observations)
        self.r, self.s_next = self.build_dynamics(latent_state, self.action_tensor)
        self.pi, self.v = self.build_predictor(latent_state)

        self.encoder = Model(inputs=self.observation_history, outputs=self.s, name="r")
        self.dynamics = Model(inputs=[self.latent_state, self.action_tensor], outputs=[self.r, self.s_next], name='d')
        self.predictor = Model(inputs=self.latent_state, outputs=[self.pi, self.v], name='p')

        self.forward = Model(inputs=self.observation_history, outputs=[self.s, *self.predictor(self.s)], name='initial')
        self.recurrent = Model(inputs=[self.latent_state, self.action_tensor],
                               outputs=[self.r, self.s_next, *self.predictor(self.s_next)], name='recurrent')

    def dense_sequence(self, n, x):  # Recursively builds a Fully Connected sequence of length n.
        if n > 0:
            return self.dense_sequence(n - 1, Activation(self.args.dense_activation)(Dense(self.args.size_dense)(x)))
        return x

    def build_encoder(self, observations):
        fc_sequence = self.dense_sequence(self.args.num_dense, observations)

        s_fc_latent = Dense(self.latents, activation='linear', name='s_0')(fc_sequence)
        latent_state = MinMaxScaler()(s_fc_latent)
        latent_state = Reshape((self.latents, 1))(latent_state)

        return latent_state  # 2-dimensional 1-time step latent state. (Encodes history of images into one state).

    def build_dynamics(self, encoded_state, action_plane):
        stacked = Concatenate()([encoded_state, action_plane])
        fc_sequence = self.dense_sequence(self.args.num_dense, stacked)

        s_fc_latent = Dense(self.latents, activation='linear', name='s_next')(fc_sequence)
        latent_state = MinMaxScaler()(s_fc_latent)
        latent_state = Reshape((self.latents, 1))(latent_state)

        r = Dense(1, activation='linear', name='r')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='r')(fc_sequence)

        return r, latent_state

    def build_predictor(self, latent_state):
        fc_sequence = self.dense_sequence(self.args.num_dense, latent_state)

        pi = Dense(self.action_size, activation='softmax', name='pi')(fc_sequence)
        v = Dense(1, activation='linear', name='v')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc_sequence)

        return pi, v
