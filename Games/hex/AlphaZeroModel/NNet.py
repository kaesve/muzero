import os
# import time
import numpy as np
import sys

from AlphaZero.NeuralNet import NeuralNet
from .HexNNet import HexNNet as onnet

sys.path.append('../../..')


class NNetWrapper(NeuralNet):
    def __init__(self, game, net_args):
        super().__init__(game)
        self.net_args = net_args
        self.nnet = onnet(game, net_args)
        self.board_x, self.board_y = game.getDimensions()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=self.net_args.batch_size,
                            epochs=self.net_args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No AlphaZeroModel in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
