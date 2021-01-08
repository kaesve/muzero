"""
This file defines helper functions for computing and formatting varying parts of the loss computation/ output
representation of the MuZero or AlphaZero neural networks.
"""
import typing

import numpy as np
import tensorflow as tf


def safe_l2norm(x, epsilon=1e-5, axis=None):
    """ Compute L2-Norm with an epsilon term for numerical stability (TODO Open github issue for this?) """
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)


def scale_gradient(tensor: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Scale gradients for reverse differentiation proportional to the given scale.
    Does not influence the magnitude/ scale of the output from a given tensor (just the gradient).
    """
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def scalar_loss(prediction: typing.Union[tf.Tensor, np.ndarray],
                target: typing.Union[tf.Tensor, np.ndarray]) -> typing.Union[tf.Tensor, float]:
    """
    Wrapper function to infer the correct loss function given the output representation.
    :param prediction: tf.Tensor or np.ndarray Output of a neural network.
    :param target: tf.Tensor or np.ndarray Target output for a neural network.
    :return: tf.losses Loss function between target and prediction
    """
    if np.prod(prediction.shape) == prediction.shape[0]:           # Implies (batch_size, 1) --> Regression
        return tf.losses.mean_squared_error(target, prediction)

    return tf.losses.categorical_crossentropy(target, prediction)  # Default: Cross Entropy


def cast_to_tensor(x: typing.Union[np.ndarray, float]) -> tf.Tensor:
    """ Wrapper function to cast numpy arrays to tensorflow Tensors with keras float default. """
    return tf.convert_to_tensor(x, dtype=tf.keras.backend.floatx())


def atari_reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
    """
    Scalar transformation of rewards to stabilize variance and reduce scale.
    :references: https://arxiv.org/pdf/1805.11593.pdf
    """
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + var_eps * x


def inverse_atari_reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
    """
    Inverse scalar transformation of atari_reward_transform function as used in the canonical MuZero paper.
    :references: https://arxiv.org/pdf/1805.11593.pdf
    """
    return np.sign(x) * (((np.sqrt(1 + 4 * var_eps * (np.abs(x) + 1 + var_eps)) - 1) / (2 * var_eps)) ** 2 - 1)


def support_to_scalar(x: np.ndarray, support_size: int,
                      inv_reward_transformer: typing.Callable = inverse_atari_reward_transform, **kwargs) -> np.ndarray:
    """
    Recast distributional representation of floats back to floats. As the bins are symmetrically oriented around 0,
    this is simply done by taking a dot-product of the vector that represents the bins' integer range with the
    probability bins. After recasting of the floats, the floats are inverse-transformed for the scaling function.
    :param x: np.ndarray 2D-array of floats in distributional representation: len(scalars) x (support_size * 2 + 1)
    :param support_size:
    :param inv_reward_transformer: Inverse of the elementwise function that scales floats before casting them to bins.
    :param kwargs: Keyword arguments for inv_reward_transformer.
    :return: np.ndarray of size len(scalars) x 1
    """
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    bins = np.arange(-support_size, support_size + 1)
    y = np.dot(x, bins)

    value = inv_reward_transformer(y, **kwargs)

    return value


def scalar_to_support(x: np.ndarray, support_size: int,
                      reward_transformer: typing.Callable = atari_reward_transform, **kwargs) -> np.ndarray:
    """
    Cast a scalar or array of scalars to a distributional representation symmetric around 0.
    For example, the float 3.4 given a support size of 5 will create 11 bins for integers [-5, ..., 5].
    Each bin is assigned a probability value of 0, bins 4 and 3 will receive probabilities .4 and .6 respectively.
    :param x: np.ndarray 1D-array of floats to be cast to distributional bins.
    :param support_size: int Number of bins indicating integer range symmetric around zero.
    :param reward_transformer: Elementwise function to scale floats before casting them to bins.
    :param kwargs: Keyword arguments for reward_transformer.
    :return: np.ndarray of size len(x) x (support_size * 2 + 1)
    """
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    # Clip float to fit within the support_size. Values exceeding this will be assigned to the closest bin.
    transformed = np.clip(reward_transformer(x, **kwargs), a_min=-support_size, a_max=support_size - 1e-8)
    floored = np.floor(transformed).astype(int)  # Lower-bound support integer
    prob = transformed - floored                 # Proportion between adjacent integers

    bins = np.zeros((len(x), 2 * support_size + 1))

    bins[np.arange(len(x)), floored + support_size] = 1 - prob
    bins[np.arange(len(x)), floored + support_size + 1] = prob

    return bins
