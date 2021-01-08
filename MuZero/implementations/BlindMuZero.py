"""
Defines an override to DefaultMuZero to use MuZero with sparse observations. This is done by 'blindfolding' the
algorithm and refreshing the observation state every 'n' steps. In other words, this class attempts to solve
the MuZero internal/ abstract MDP and gets reset every 'n' steps towards a new starting state. Following the paradigm
of value-equivalence methods, if the value/ reward space of the internal/ abstract MDP is equivalent to the
environment MDP, this will in parallel solve both MDPs.

This class should be able to handle most MuZero neural network implementations on most environment.

Notes:
 -  Base implementation done 29/11/2020
 -  Documentation 29/11/2020
TODO: Compatibility MuCoach/ training.
"""
import typing

import numpy as np

from utils import DotDict
from utils.selfplay_utils import GameHistory
from .DefaultMuZero import DefaultMuZero


class BlindMuZero(DefaultMuZero):
    """
    DefaultMuZero implementation to plan/ act in an environment with sparse observations.
    """

    def __init__(self, game, net_args: DotDict, architecture: str, refresh_freq: int = 1) -> None:
        """
        Initialize the MuZero Neural Network. Selects a Neural Network class constructor by an 'architecture' string.
        See Agents/MuZeroNetworks for all available neural architectures, or for defining new architectures.

        Addition: refresh_freq governs the frequency with which state observations get embedded into the neural network.
        :param game: Implementation of base Game class for environment logic.
        :param net_args: DotDict Data structure that contains all neural network arguments as object attributes.
        :param architecture: str Neural network architecture to build in the super class.
        :param refresh_freq: int Frequency of embedding state observations in the neural network's MDP.
        """
        super().__init__(game, net_args, architecture)
        self.refresh_freq = refresh_freq

        self.action_reference = list()
        self.memory = None  # Most recent RNN neural activations (parallel to time with the agent's real environment).
        self.steps = 0

    def bind(self, action_reference: GameHistory) -> None:
        """ Bind a list reference of actions to this class to perform forward steps within the embedded space. """
        self.action_reference = action_reference

    def reset(self) -> None:
        """
        Reset the class embedded state variable and step count
        """
        self.memory = None
        self.steps = 0

    def initial_inference(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float]:
        """
        Override base initial_inference to only embed state observations every `self.refresh_freq' steps.
        Otherwise, it uses the last performed action to take a step forward in the embedded space.
        :param observations: A game specific (stacked) tensor of observations of the environment at step t: o_t.
        :return: A tuple with predictions of the following form:
            s_(0): The root 'latent_state' produced by the representation function
            pi: a policy vector for the provided state - a numpy array of length |action_space|.
            v: a float that gives the state value estimate of the provided state.
        :see: DefaultMuZero
        """
        if self.memory is None or self.steps % self.refresh_freq == 0:
            # Refresh
            s, pi, v = super().initial_inference(observations)
        else:
            _, s, pi, v = super().recurrent_inference(self.memory, self.action_reference[-1])

        self.memory = s
        self.steps += 1

        return s, pi, v

