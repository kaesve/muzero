"""

"""

import sys

from keras.models import Model
from keras.layers import Input, Reshape, Activation, Dense, Conv2D, \
    AveragePooling2D, BatchNormalization, Concatenate, Flatten
from keras.optimizers import Adam

from utils.network_utils import MinMaxScaler, Crafter

sys.path.append('../..')


class AlphaZeroAtariNetwork:

    def __init__(self, game, args):
        self.board_x, self.board_y, depth = game.getDimensions()
        self.action_size = game.getActionSize()
        self.args = args
        self.crafter = Crafter(args)

        # Neural Net# s: batch_size x board_x x board_y
        self.input_boards = Input(shape=(self.board_x, self.board_y, depth * self.args.observation_length))

        self.pi, self.v = self.build_model(self.input_boards)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])

        opt = Adam(args.optimizer.lr_init)
        if self.args.support_size > 0:
            self.model.compile(loss=['categorical_crossentropy'] * 2, optimizer=opt)
        else:
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=opt)

        print(self.model.summary())

    def build_model(self, x_image):
        conv = Conv2D(self.args.num_channels, kernel_size=3, strides=(2, 2))(x_image)
        res = self.crafter.conv_residual_tower(self.args.num_towers, conv,
                                               self.args.residual_left, self.args.residual_right)

        conv = Conv2D(self.args.num_channels, kernel_size=3, strides=(2, 2))(res)
        res = self.crafter.conv_residual_tower(self.args.num_towers, conv,
                                               self.args.residual_left, self.args.residual_right)

        pooled = AveragePooling2D(3, strides=(2, 2))(res)
        res = self.crafter.conv_residual_tower(self.args.num_towers, pooled,
                                               self.args.residual_left, self.args.residual_right)

        pooled = AveragePooling2D(3, strides=(2, 2))(res)
        conv = Conv2D(self.args.num_channels // 2, kernel_size=3, strides=(2, 2))(pooled)
        flat = Flatten()(conv)

        fc = self.crafter.dense_sequence(1, flat)

        pi = Dense(self.action_size, activation='softmax', name='pi')(fc)
        v = Dense(1, activation='tanh', name='v')(fc) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc)

        return pi, v


class MuZeroAtariNetwork:

    def __init__(self, game, args):
        # Network arguments
        self.board_x, self.board_y, self.planes = game.getDimensions()
        self.latent_x, self.latent_y = (6, 6)
        self.action_size = game.getActionSize()
        self.args = args
        self.crafter = Crafter(args)

        assert self.action_size == self.latent_x * self.latent_y, \
            "The action space should be the same size as the latent space"

        # s: batch_size x time x state_x x state_y
        self.observation_history = Input(shape=(self.board_x, self.board_y, self.planes * self.args.observation_length))
        # a: one hot encoded vector of shape batch_size x (state_x * state_y)
        self.action_plane = Input(shape=(self.action_size,))
        # s': batch_size  x board_x x board_y x 1
        self.latent_state = Input(shape=(self.latent_x, self.latent_y, 1))

        action_plane = Reshape((self.latent_x, self.latent_y, 1))(self.action_plane)
        latent_state = Reshape((self.latent_x, self.latent_y, 1))(self.latent_state)

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
        down_sampled = Activation('relu')(BatchNormalization()(Conv2D(64, 3, 2)(observations)))
        down_sampled = Activation('relu')(BatchNormalization()(Conv2D(128, 3, 2)(down_sampled)))
        down_sampled = AveragePooling2D(3, 2)(down_sampled)
        down_sampled = AveragePooling2D(3, 2)(down_sampled)

        out_tensor = self.crafter.build_conv_block(down_sampled)

        s_fc_latent = Dense(self.latent_x * self.latent_y, activation='linear', name='s_0')(out_tensor)
        latent_state = MinMaxScaler()(s_fc_latent)
        latent_state = Reshape((self.latent_x, self.latent_y, 1))(latent_state)

        return latent_state  # 2-dimensional 1-time step latent state. (Encodes history of images into one state).

    def build_dynamics(self, encoded_state, action_plane):
        stacked = Concatenate(axis=-1)([encoded_state, action_plane])
        reshaped = Reshape((self.latent_x, self.latent_y, -1))(stacked)
        out_tensor = self.crafter.build_conv_block(reshaped)

        s_fc_latent = Dense(self.latent_x * self.latent_y, activation='linear', name='s_next')(out_tensor)
        latent_state = Reshape((self.latent_x, self.latent_y, 1))(s_fc_latent)
        latent_state = MinMaxScaler()(latent_state)

        r = Dense(1, activation='linear', name='r')(out_tensor) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='r')(out_tensor)

        return r, latent_state

    def build_predictor(self, latent_state):
        out_tensor = self.crafter.build_conv_block(latent_state)

        pi = Dense(self.action_size, activation='softmax', name='pi')(out_tensor)
        v = Dense(1, activation='tanh', name='v')(out_tensor) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(out_tensor)

        return pi, v
