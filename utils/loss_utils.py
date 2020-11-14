"""

"""
import typing

import numpy as np
import tensorflow as tf


def check_nans(list_of_tensors: typing.List) -> bool:
    return np.any([np.any(np.isnan(x)) for x in list_of_tensors])


def safe_l2norm(x, epsilon=1e-5, axis=None):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)


def scale_gradient(tensor: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Scales the gradient for the backward pass.
    :param tensor:
    :param scale:
    :return:
    """
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def scalar_loss(prediction: typing.Union[tf.Tensor, np.ndarray],
                target: typing.Union[tf.Tensor, np.ndarray]) -> typing.Union[tf.Tensor, float]:
    """

    :param prediction:
    :param target:
    :return:
    """
    if np.prod(prediction.shape) == prediction.shape[0]:           # Implies (batch_size, 1) --> Regression
        return tf.losses.mean_squared_error(target, prediction)

    return tf.losses.categorical_crossentropy(target, prediction)  # Default: Cross Entropy


def cast_to_tensor(x: typing.Union[np.ndarray, float]) -> tf.Tensor:
    """

    :param x:
    :return:
    """
    return tf.convert_to_tensor(x, dtype=tf.keras.backend.floatx())


def atari_reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
    """

    :param x:
    :param var_eps:
    :return:
    :references: https://arxiv.org/pdf/1805.11593.pdf
    """
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + var_eps * x


def inverse_atari_reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
    """

    :param x:
    :param var_eps:
    :return:
    :references: https://arxiv.org/pdf/1805.11593.pdf
    """
    return np.sign(x) * (((np.sqrt(1 + 4 * var_eps * (np.abs(x) + 1 + var_eps)) - 1) / (2 * var_eps)) ** 2 - 1)


def support_to_scalar(x: np.ndarray, support_size: int,
                      reward_transformer: typing.Callable = inverse_atari_reward_transform, **kwargs) -> np.ndarray:
    """

    :param x:
    :param support_size:
    :param reward_transformer:
    :param kwargs:
    :return:
    """
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    bins = np.arange(-support_size, support_size + 1)
    y = np.dot(x, bins)

    value = reward_transformer(y, **kwargs)

    return value


def scalar_to_support(x: np.ndarray, support_size: int,
                      reward_transformer: typing.Callable = atari_reward_transform, **kwargs) -> np.ndarray:
    """

    :param x:
    :param support_size:
    :param reward_transformer:
    :param kwargs:
    :return:
    """
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    # Reduce the scale
    transformed = np.clip(reward_transformer(x, **kwargs), a_min=-support_size, a_max=support_size)
    floored = np.floor(transformed).astype(int)  # Lower-bound support integer
    prob = transformed - floored                 # Proportion between adjacent integers

    bins = np.zeros((len(x), 2 * support_size + 1))

    bins[np.arange(len(x)), floored + support_size] = 1 - prob
    bins[np.arange(len(x)), floored + support_size + 1] = prob

    return bins
