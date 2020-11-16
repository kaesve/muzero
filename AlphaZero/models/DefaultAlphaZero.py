"""
Defines a generic implementation of AlphaNeuralNet to handle data for training, and performing standard inference.

This class should be able to handle most AlphaZero neural network models on most environment.

Notes:
 -  Base implementation done 15/11/2020
 -  Documentation 16/11/2020
"""
import sys
import typing

import numpy as np
import tensorflow as tf

from AlphaZero.AlphaNeuralNet import AlphaZeroNeuralNet
from utils.loss_utils import support_to_scalar, scalar_to_support
from utils import DotDict
import Agents

sys.path.append('../../..')


class DefaultAlphaZero(AlphaZeroNeuralNet):
    """
    This class implements the base AlphaZeroNeuralNet class. The implemented methods perform data manipulation to train
    the AlphaZero agent and return inferred variables given the necessary inputs. This implementation should work
    for most neural architectures and provided environments.
    """

    def __init__(self, game, net_args: DotDict, architecture: str) -> None:
        """
        Initialize the AlphaZero Neural Network. Selects a Neural Network class constructor by an 'architecture' string.
        See Agents/AlphaZeroNetworks for all available neural architectures, or for defining new architectures.
        :param game: Implementation of base Game class for environment logic.
        :param net_args: DotDict Data structure that contains all neural network arguments as object attributes.
        :param architecture: str Neural network architecture to build in the super class.
        """
        super().__init__(game, net_args, Agents.AlphaZeroNetworks[architecture])
        self.action_size = game.getActionSize()
        self.architecture = architecture

    def train(self, examples: typing.List) -> None:
        """
        This function trains the neural network with data gathered from self-play.

        We unpack the data contained in the provided list (of length batch_size) and cast the variables to np.ndarrays.
        The target values are cast to distribution bins based on the neural network arguments.

        The method performs one step of gradient descent entirely within the keras model given data and sample weights.
        Loss values and weight norm are recorded in the class Monitor.

        :param examples: a list of training examples of the form: (o_t, (pi_t, v_t), w_t)
        """
        observations, targets, loss_scale = list(zip(*examples))
        target_pis, target_vs = list(map(np.asarray, zip(*targets)))

        # ```np.asarray``` does not copy data contained within iterable
        observations = np.asarray(observations)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        priorities = np.asarray(loss_scale)

        # Cast to distribution
        target_vs = scalar_to_support(target_vs, self.net_args.support_size)

        total_loss, pi_loss, v_loss = self.neural_net.model.train_on_batch(
            x=observations, y=[target_pis, target_vs], sample_weight=[priorities, priorities], reset_metrics=True)
        l2_norm = tf.reduce_sum([tf.nn.l2_loss(x) for x in self.neural_net.model.get_weights()])

        self.monitor.log(pi_loss, "pi_loss")
        self.monitor.log(v_loss, "v_loss")

        self.monitor.log(total_loss, "total loss")
        self.monitor.log(l2_norm, "l2 norm")

        self.steps += 1

    def predict(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, float]:
        """
        Infer the neural network move probability prior and state value given a state observation.

        The observation array is padded with a batch-size dimension of length 1. The inferred state value is
        cast from its distributional bins into a scalar.

        :param observations: Observation representation of the form (width x height x (depth * time)
        :return: A tuple with predictions of the following form:
            pi: a policy vector for the provided state - a numpy array of length |action_space|.
            v: a float that gives the state value estimate of the provided state.
        """
        # Pad input with batch dimension
        observation = observations[np.newaxis, ...]

        pi, v = self.neural_net.model.predict(observation)

        # Cast distribution bins to scalar
        v_real = support_to_scalar(v, self.net_args.support_size)

        return pi[0], np.ndarray.item(v_real)
