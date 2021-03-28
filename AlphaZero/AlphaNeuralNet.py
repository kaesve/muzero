"""
Defines the base structure of a neural network for AlphaZero. Loss computation/ gradient optimization relies
on the backend of tensorflow/ keras.

Notes:
 - Base implementation done.
 - Documentation 14/11/2020
"""
import typing
import os
from abc import ABC, abstractmethod

import numpy as np

from utils.debugging import AlphaZeroMonitor
from utils import DotDict


class AlphaZeroNeuralNet(ABC):
    """
    This class specifies the base AlphaZeroNeuralNet class. To define your own neural network, subclass
    this class and implement the abstract functions below. See DefaultAlphaZero for an example implementation.
    """

    def __init__(self, game, net_args: DotDict, builder: typing.Callable) -> None:
        """
        Initialize base AlphaZero Neural Network. Contains all requisite logic to work with any AlphaZero
         network and environment.
        :param game: Implementation of base Game class for environment logic.
        :param net_args: DotDict Data structure that contains all neural network arguments as object attributes.
        :param builder: Function that takes the game and network arguments as parameters and returns a tf.keras.Model
        """
        self.single_player = (game.n_players == 1)
        self.net_args = net_args
        self.neural_net = builder(game, net_args)
        self.monitor = AlphaZeroMonitor(self)

        # Reference variable for tracking the number of gradient update steps.
        self.steps = 0

    @abstractmethod
    def train(self, examples: typing.List) -> None:
        """
        This function trains the neural network with data gathered from self-play.

        :param examples: a list of training examples of the form: (o_t, (pi_t, v_t), w_t)
        """

    @abstractmethod
    def predict(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, float]:
        """
        Infer the neural network move probability prior and state value given a state observation.

        :param observations: Observation representation of the form (width x height x (depth * time)
        :return: A tuple with predictions of the following form:
            pi: a policy vector for the provided state - a numpy array of length |action_space|.
            v: a float that gives the state value estimate of the provided state.
        """

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """
        Saves the current neural network (with its parameters) in folder/filename
        If specified folder does not yet exists, the method creates a new folder if permitted.
        :param folder: str Path to model weight file
        :param filename: str Base name for model weight file
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.neural_net.model.save_weights(filepath)

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """
        Loads parameters of the neural network model from the given folder/filename

        :param folder: str Path to model weight file
        :param filename: str Base name of model weight file
        :raises: FileNotFoundError if path is incorrectly specified.
        """
        filepath = os.path.join(folder, filename)
        try:
            self.neural_net.model.load_weights(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"No AlphaZero Model in path {filepath}")
