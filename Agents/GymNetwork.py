"""
Defines default MLP neural networks for both AlphaZero and MuZero that can be used on simple classical control
environments, such as those from OpenAI Gym.
"""

import sys

from keras.layers import Dense, Input, Reshape, Concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam

from utils.network_utils import Crafter, MinMaxScaler

sys.path.append('../..')


class AlphaZeroGymNetwork:

    def __init__(self, game, args):
        # Network arguments
        self.x, self.y, self.planes = game.getDimensions()

        self.action_size = game.getActionSize()
        self.args = args
        self.crafter = Crafter(args)

        # s: batch_size x time x state_x x state_y
        self.observation_history = Input(shape=(self.x, self.y, self.planes * self.args.observation_length))
        # a: one hot encoded vector of shape batch_size x (state_x * state_y)
        self.action_tensor = Input(shape=(self.action_size, ))

        observations = Reshape((self.x * self.y * self.planes * self.args.observation_length, ))(
            self.observation_history)

        self.pi, self.v = self.build_predictor(observations)

        self.model = Model(inputs=self.observation_history, outputs=[self.pi, self.v])

        opt = Adam(args.optimizer.lr_init)
        if self.args.support_size > 0:
            self.model.compile(loss=['categorical_crossentropy'] * 2, optimizer=opt)
        else:
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=opt)

    def build_predictor(self, observations):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, observations)

        pi = Dense(self.action_size, activation='softmax', name='pi')(fc_sequence)
        v = Dense(1, activation='linear', name='v')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc_sequence)

        return pi, v


class MuZeroGymNetwork:

    def __init__(self, game, args):
        # Network arguments
        self.x, self.y, self.planes = game.getDimensions()
        self.latents = args.latent_depth
        self.action_size = game.getActionSize()
        self.args = args
        self.crafter = Crafter(args)

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

        # Decoder functionality.
        self.decoded_observations = self.build_decoder(latent_state)
        self.decoder = Model(inputs=self.latent_state, outputs=self.decoded_observations, name='decoder')

    def build_encoder(self, observations):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, observations)

        latent_state = Dense(self.latents, activation='linear', name='s_0')(fc_sequence)
        latent_state = Activation('tanh')(latent_state) if self.latents <= 3 else MinMaxScaler()(latent_state)
        latent_state = Reshape((self.latents, 1))(latent_state)

        return latent_state  # 2-dimensional 1-time step latent state. (Encodes history of images into one state).

    def build_dynamics(self, encoded_state, action_plane):
        stacked = Concatenate()([encoded_state, action_plane])
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, stacked)

        latent_state = Dense(self.latents, activation='linear', name='s_next')(fc_sequence)
        latent_state = Activation('tanh')(latent_state) if self.latents <= 3 else MinMaxScaler()(latent_state)
        latent_state = Reshape((self.latents, 1))(latent_state)

        r = Dense(1, activation='linear', name='r')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='r')(fc_sequence)

        return r, latent_state

    def build_predictor(self, latent_state):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, latent_state)

        pi = Dense(self.action_size, activation='softmax', name='pi')(fc_sequence)
        v = Dense(1, activation='linear', name='v')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc_sequence)

        return pi, v

    def build_decoder(self, latent_state):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, latent_state)

        out = Dense(self.x * self.y * self.planes, name='o_k')(fc_sequence)
        o = Reshape((self.x, self.y, self.planes))(out)
        return o

