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

sys.path.append('..')


class HexNNet:

    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y, depth = game.getDimensions()
        self.action_size = game.getActionSize()
        self.args = args
        self.crafter = Crafter(args)

        # Neural Net# s: batch_size x board_x x board_y
        self.input_boards = Input(shape=(self.board_x, self.board_y, depth * self.args.observation_length))

        self.pi, self.v = self.build_model(self.input_boards)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        if self.args.support_size > 0:
            self.model.compile(loss=['categorical_crossentropy'] * 2, optimizer=Adam(args.lr))
        else:
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
        print(self.model.summary())

    def build_model(self, x_image):
        conv = self.crafter.conv_tower(self.args.num_convs, x_image)
        res = self.crafter.conv_residual_tower(self.args.num_towers, conv,
                                               self.args.residual_left, self.args.residual_right)

        small = self.crafter.activation()(BatchNormalization()(Conv2D(32, 3, padding='same', use_bias=False)(res)))

        flat = Flatten()(small)

        fc = self.crafter.dense_sequence(1, flat)

        pi = Dense(self.action_size, activation='softmax', name='pi')(fc)
        v = Dense(1, activation='tanh', name='v')(fc) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc)

        return pi, v
