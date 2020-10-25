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
        self.board_x, self.board_y = game.getDimensions()
        self.action_size = game.getActionSize()
        self.args = args
        self.crafter = Crafter(args)

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))  # s: batch_size x board_x x board_y
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)  # batch_size  x board_x x board_y x 1

        self.pi, self.v = self.build_model(x_image)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        if self.args.support_size > 0:
            self.model.compile(loss=['categorical_crossentropy'] * 2, optimizer=Adam(args.lr))
        else:
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))

    def build_model(self, x_image):
        h_conv3 = self.crafter.conv_tower(3, x_image)
        h_conv3_flat = Flatten()(h_conv3)
        s_fc1 = Dropout(self.args.dropout)(Activation('relu')(Dense(512)(h_conv3_flat)))  # Widened

        pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc1)
        v = Dense(1, activation='tanh', name='v')(s_fc1) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(s_fc1)

        return pi, v
