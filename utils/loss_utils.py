import numpy as np
import tensorflow as tf


def scalar_loss(prediction, target):
    if np.prod(prediction.shape) == len(prediction):                    # Implies (batch_size, 1) --> Regression
        return tf.losses.mse(prediction - target)                       # MSE

    return tf.reduce_sum(target * tf.math.log(prediction), axis=1)      # Default: Cross Entropy


def atari_reward_transform(x, var_eps=0.001):
    # See https://arxiv.org/pdf/1805.11593.pdf
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + var_eps * x


def inverse_atari_reward_transform(x, var_eps=0.001):
    # See https://arxiv.org/pdf/1805.11593.pdf
    return np.sign(x) * (((np.sqrt(1 + 4 * var_eps * (np.abs(x) + 1 + var_eps)) - 1) / (2 * var_eps)) ** 2 - 1)


def support_to_scalar(x: np.ndarray, support_size: int, reward_transformer=inverse_atari_reward_transform, **kwargs):
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    bins = np.arange(-support_size, support_size + 1)
    y = x.dot(bins)

    value = reward_transformer(y, **kwargs)

    return value


def scalar_to_support(x: np.ndarray, support_size: int, reward_transformer=atari_reward_transform, **kwargs):
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    # Reduce the scale
    transformed = reward_transformer(x, **kwargs)
    floored = np.floor(transformed).astype(int)  # Lower-bound support integer
    prob = transformed - floored     # Proportion between adjacent integers

    bins = np.zeros((len(x), 2 * support_size + 1))

    bins[np.arange(len(x)), floored + support_size] = 1 - prob
    bins[np.arange(len(x)), floored + support_size + 1] = prob

    return bins


if __name__ == "__main__":
    scalars = np.arange(10)
    support = scalar_to_support(scalars, 20)
    inverted = support_to_scalar(support, 20)

    assert np.linalg.norm(scalars - inverted) < 1e-8, "x != f^-1(f(x))"



