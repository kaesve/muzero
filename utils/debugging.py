import logging
import typing

# Bug fixing TF2?
# Prevent TF2 from hogging all the available VRAM when initializing?
# @url: https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf
import numpy as np

from utils.loss_utils import scale_gradient, scalar_loss

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Bug fixing TF2?

# Suppress warnings from TENSORFLOW's side
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


class MuZeroMonitor:

    def __init__(self, reference, logging_args=None):  # TODO Logging args integration.
        self.reference = reference
        self.logging_args = logging_args

    def log_recurrent_losses(self, t: int, scale: tf.Tensor, v_loss: tf.Tensor,
                             r_loss: tf.Tensor, pi_loss: tf.Tensor) -> None:
        tf.summary.scalar(f"r_loss_{t}", data=tf.reduce_sum(scale_gradient(r_loss, scale)), step=self.reference.steps)
        tf.summary.scalar(f"v_loss_{t}", data=tf.reduce_sum(scale_gradient(v_loss, scale)), step=self.reference.steps)
        tf.summary.scalar(f"pi_loss_{t}", data=tf.reduce_sum(scale_gradient(pi_loss, scale)), step=self.reference.steps)

    def log(self, tensor: typing.Union[tf.Tensor, float], name: str) -> None:
        tf.summary.scalar(name, data=tensor, step=self.reference.steps)

    def log_distribution(self, tensor: typing.Union[tf.Tensor, np.ndarray], name: str) -> None:
        tf.summary.histogram(name, tensor, step=self.reference.steps)

    def log_batch(self, data_batch) -> None:  # TODO
        pass


class AlphaZeroMonitor:

    def __init__(self, reference, logging_args=None):  # TODO Logging args integration.
        self.reference = reference
        self.logging_args = logging_args

    def log(self, tensor: typing.Union[tf.Tensor, float], name: str) -> None:
        tf.summary.scalar(name, data=tensor, step=self.reference.steps)

    def log_distribution(self, tensor: typing.Union[tf.Tensor, np.ndarray], name: str) -> None:
        tf.summary.histogram(name, tensor, step=self.reference.steps)
