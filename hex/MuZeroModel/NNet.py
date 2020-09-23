import os
import numpy as np
import sys

from MuZero.MuNeuralNet import MuZeroNeuralNet
from .HexNNet import HexNNet as NetBuilder

sys.path.append('../..')


class NNetWrapper(MuZeroNeuralNet):
    def __init__(self, game, net_args):
        super().__init__(game)
        self.net_args = net_args
        self.neural_net = NetBuilder(game, net_args)
        self.board_x, self.board_y = game.getDimensions()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        """
        pass

    def encode(self, observations):
        observations = observations[np.newaxis, ...]
        print(observations.shape)
        return self.neural_net.encoder.predict(observations)[0]

    def forward(self, latent_state, action):
        a_plane = np.zeros((self.board_x, self.board_y))
        a_plane[action // self.board_x][action % self.board_y] = 1

        latent_state = latent_state.reshape((-1, self.board_x, self.board_y))
        a_plane = a_plane.reshape((-1, self.board_x, self.board_y))

        r, s_next = self.neural_net.dynamics.predict([latent_state, a_plane])
        return r[0], s_next[0]

    def predict(self, latent_state):
        """
        board: np array with board
        """
        latent_state = latent_state.reshape((-1, self.board_x, self.board_y))
        pi, v = self.neural_net.predictor.predict(latent_state)
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.neural_net.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No AlphaZeroModel in path {}".format(filepath))
        self.neural_net.model.load_weights(filepath)
