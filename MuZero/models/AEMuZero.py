import numpy as np
import sys
import typing

import tensorflow as tf
from tensorflow import GradientTape

from utils.loss_utils import support_to_scalar, scalar_to_support, cast_to_tensor
from MuZero.models.DefaultMuZero import DefaultMuZero
from .architectures import *

from utils.storage import DotDict

sys.path.append('../../..')

models = {
    "Gym": BuildGymNet,
    "Hex": BuildHexNet,
    "Atari": BuildAtariNet
}


class StabilizedMuZero(DefaultMuZero):

    def train(self, examples: typing.List) -> float:
        return super().train(examples)

    def loss_function(self, observations: tf.Tensor, actions: tf.Tensor, target_vs: tf.Tensor, target_rs: tf.Tensor,
                      target_pis: tf.Tensor, sample_weights: tf.Tensor) -> typing.Tuple[tf.Tensor, typing.List]:
        return super().loss_function(observations, actions, target_vs, target_rs, target_pis, sample_weights)
