"""
In this python file we modified the class to enable generation of multiple
neural architectures by modifying the argument dictionary.

For the details of the neural architectures, we refer to our report.
:see: main_experimenter.py
"""

import sys

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

from utils.network_utils import Crafter

sys.path.append('../..')


class GymNNet:

    def __init__(self, game, args):
        # Network arguments
        self.x, self.y, self.planes = game.getDimensions()

        self.action_size = game.getActionSize()
        self.args = args
        self.crafter = Crafter(args)

        # s: batch_size x time x state_x x state_y
        self.observation_history = Input(shape=(self.x, self.y, self.planes))
        # a: one hot encoded vector of shape batch_size x (state_x * state_y)
        self.action_tensor = Input(shape=(self.action_size, ))

        observations = Reshape((self.x * self.y * self.planes, ))(self.observation_history)

        self.pi, self.v = self.build_predictor(observations)

        self.model = Model(inputs=self.observation_history, outputs=[self.pi, self.v])
        if self.args.support_size > 0:
            self.model.compile(loss=['categorical_crossentropy'] * 2, optimizer=Adam(args.lr))
        else:
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))

    def build_predictor(self, observations):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, observations)

        pi = Dense(self.action_size, activation='softmax', name='pi')(fc_sequence)
        v = Dense(1, activation='linear', name='v')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc_sequence)

        return pi, v
