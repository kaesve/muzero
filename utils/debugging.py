import logging
import typing

# Dynamic VRAM growth: https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf
import numpy as np

from utils.loss_utils import scale_gradient, support_to_scalar, scalar_to_support

# Dynamic VRAM growth: https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Suppress warnings from TENSORFLOW's side
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
tf.autograph.set_verbosity(1)

DEBUG_MODE = False
LOG_RATE = 1


class Monitor:

    def __init__(self, reference):
        self.reference = reference

    def log(self, tensor: typing.Union[tf.Tensor, float], name: str) -> None:
        if self.reference.steps % LOG_RATE == 0:
            tf.summary.scalar(name, data=tensor, step=self.reference.steps)

    def log_distribution(self, tensor: typing.Union[tf.Tensor, np.ndarray], name: str) -> None:
        if self.reference.steps % LOG_RATE == 0:
            tf.summary.histogram(name, tensor, step=self.reference.steps)

    def log_batch(self, data_batch: typing.List) -> None:
        pass


class MuZeroMonitor(Monitor):

    def __init__(self, reference):
        super().__init__(reference)

    def log_recurrent_losses(self, t: int, scale: tf.Tensor, v_loss: tf.Tensor,
                             r_loss: tf.Tensor, pi_loss: tf.Tensor) -> None:
        step = self.reference.steps
        if self.reference.steps % LOG_RATE == 0:
            tf.summary.scalar(f"r_loss_{t}", data=tf.reduce_sum(scale_gradient(r_loss, scale)), step=step)
            tf.summary.scalar(f"v_loss_{t}", data=tf.reduce_sum(scale_gradient(v_loss, scale)), step=step)
            tf.summary.scalar(f"pi_loss_{t}", data=tf.reduce_sum(scale_gradient(pi_loss, scale)), step=step)

    def log_batch(self, data_batch: typing.List) -> None:
        if DEBUG_MODE and self.reference.steps % LOG_RATE == 0:
            observations, actions, targets, sample_weight = list(zip(*data_batch))
            actions, sample_weight = np.asarray(actions), np.asarray(sample_weight)
            target_vs, target_rs, target_pis = list(map(np.asarray, zip(*targets)))

            priority = sample_weight * len(data_batch)  # Undo 1/n scaling to get priority
            tf.summary.histogram(f"sample probability", data=priority, step=self.reference.steps)

            s, pi, v = self.reference.neural_net.forward.predict_on_batch(np.asarray(observations))

            v_real = support_to_scalar(v, self.reference.net_args.support_size)

            tf.summary.histogram(f"v_predict_{0}", data=v_real, step=self.reference.steps)
            tf.summary.histogram(f"v_target_{0}", data=target_vs[:, 0], step=self.reference.steps)
            tf.summary.scalar(f"v_mse_{0}", data=np.mean((v_real - target_vs[:, 0]) ** 2), step=self.reference.steps)

            collect = list()
            for k in range(actions.shape[1]):
                r, s, pi, v = self.reference.neural_net.recurrent.predict_on_batch([s, actions[:, k, :]])
                collect.append(
                    (support_to_scalar(v, self.reference.net_args.support_size),
                     support_to_scalar(r, self.reference.net_args.support_size)))

            for t, (v, r) in enumerate(collect):
                k = t + 1

                tf.summary.histogram(f"r_predict_{k}", data=r, step=self.reference.steps)
                tf.summary.histogram(f"v_predict_{k}", data=v, step=self.reference.steps)

                tf.summary.histogram(f"r_target_{k}", data=target_rs[:, k], step=self.reference.steps)
                tf.summary.histogram(f"v_target_{k}", data=target_vs[:, k], step=self.reference.steps)

                tf.summary.scalar(f"r_mse_{k}", data=np.mean((r - target_rs[:, k]) ** 2), step=self.reference.steps)
                tf.summary.scalar(f"v_mse_{k}", data=np.mean((v - target_vs[:, k]) ** 2), step=self.reference.steps)


class AlphaZeroMonitor(Monitor):

    def __init__(self, reference):
        super().__init__(reference)

    def log_batch(self, data_batch: typing.List) -> None:
        if DEBUG_MODE and self.reference.steps % LOG_RATE == 0:
            observations, targets, sample_weight = list(zip(*data_batch))
            target_pis, target_vs = list(map(np.asarray, zip(*targets)))
            observations = np.asarray(observations)

            priority = sample_weight * len(data_batch)  # Undo 1/n scaling to get priority
            tf.summary.histogram(f"sample probability", data=priority, step=self.reference.steps)

            pis, vs = self.reference.neural_net.model.predict_on_batch(observations)
            v_reals = support_to_scalar(vs, self.reference.net_args.support_size)

            tf.summary.histogram(f"v_targets", data=target_vs, step=self.reference.steps)
            tf.summary.histogram(f"v_predict", data=v_reals, step=self.reference.steps)

            mse = np.mean(np.square(v_reals - target_vs))
            tf.summary.scalar("v_mse", data=mse, step=self.reference.steps)



