"""
In this python file we modified the class to enable generation of multiple
neural architectures by modifying the argument dictionary.

For the details of the neural architectures, we refer to our report.
:see: main_experimenter.py
"""

import sys
sys.path.append('..')
from utils import *

from keras.models import *
from keras.layers import *
from keras.optimizers import *


class HexNNet:

    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)  # batch_size  x board_x x board_y x 1

        if args.default:
            self.pi, self.v, self.model = self.alpha_zero_general_default(x_image, args)
        else:
            if args.transfer == 0:
                self.pi, self.v, self.model = self.shallow_model(x_image, args)
            elif args.transfer == 1:
                # add a third convolutional block and widen the fully connected layer.
                self.pi, self.v, self.model = self.medium_model(x_image, args)
            else:
                # add a fourth convolutional block and a fully connected layer and widen the 1st fc layer.
                self.pi, self.v, self.model = self.deep_model(x_image, args)

        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))

    def alpha_zero_general_default(self, x_image, args):
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv3)))  # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))  # batch_size x 1024

        pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
        v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        model = Model(inputs=self.input_boards, outputs=[pi, v])
        return pi, v, model

    def shallow_model(self, x_image, args):
        h_conv1 = Activation('relu')(BatchNormalization()(Conv2D(args.num_channels, 3, padding='same')(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization()(Conv2D(args.num_channels, 3, padding='same')(h_conv1)))
        h_conv2_flat = Flatten()(h_conv2)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(Dense(256)(h_conv2_flat)))

        pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc1)
        v = Dense(1, activation='tanh', name='v')(s_fc1)

        model = Model(inputs=self.input_boards, outputs=[pi, v])
        return pi, v, model

    def medium_model(self, x_image, args):
        h_conv1 = Activation('relu')(BatchNormalization()(Conv2D(args.num_channels, 3, padding='same')(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization()(Conv2D(args.num_channels, 3, padding='same')(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization()(Conv2D(args.num_channels, 3, padding='same')(h_conv2))) # new
        h_conv3_flat = Flatten()(h_conv3)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(Dense(512)(h_conv3_flat)))  # Widened

        pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc1)  # batch_size x self.action_size
        v = Dense(1, activation='tanh', name='v')(s_fc1)  # batch_size x 1

        model = Model(inputs=self.input_boards, outputs=[pi, v])
        return pi, v, model

    def deep_model(self, x_image, args):
        h_conv1 = Activation('relu')(BatchNormalization()(Conv2D(args.num_channels, 3, padding='same')(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization()(Conv2D(args.num_channels, 3, padding='same')(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization()(Conv2D(args.num_channels, 3, padding='same')(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization()(Conv2D(args.num_channels, 3, padding='same')(h_conv3))) # new
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(Dense(1024)(h_conv4_flat)))  # widened
        s_fc2 = Dropout(args.dropout)(Activation('relu')(Dense(512)(s_fc1))) # deepened

        pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
        v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        model = Model(inputs=self.input_boards, outputs=[pi, v])
        return pi, v, model
