import os
from abc import ABC, abstractmethod

from utils.debugging import AlphaZeroMonitor


class AlphaZeroNeuralNet(ABC):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game, net_args, builder):
        self.single_player = (game.n_players == 1)
        self.net_args = net_args
        self.neural_net = builder(game, net_args)
        self.monitor = AlphaZeroMonitor(self)

        self.steps = 0

    @abstractmethod
    def train(self, examples, steps):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        pass

    @abstractmethod
    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        pass

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """ Saves the current neural network (with its parameters) in folder/filename """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.neural_net.model.save_weights(filepath)

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """ Loads parameters of the neural network from folder/filename """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No AlphaZeroModel in path {filepath}")
        self.neural_net.model.load_weights(filepath)

