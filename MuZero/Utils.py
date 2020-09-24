import numpy as np


def atari_reward_transform(x, var_eps=0.001):
    # See https://arxiv.org/pdf/1805.11593.pdf
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + var_eps * x


def inverse_atari_reward_transform(y, var_eps=0.001):
    # See https://arxiv.org/pdf/1805.11593.pdf
    return np.sign(y) * (((np.sqrt(1 + 4 * var_eps * (np.abs(y) + 1 + var_eps))) / (2 * var_eps)) ** 2 - 1)


def support_to_scalar(logits, support_size, reward_transformer=inverse_atari_reward_transform, **kwargs):
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return logits[0]

    bins = np.arange(-support_size, support_size + 1)
    y = bins.dot(logits)

    value = reward_transformer(y, **kwargs)
    return value


def scalar_to_support(scalar, support_size, reward_transformer=atari_reward_transform, **kwargs):
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return np.array([scalar])

    # Reduce the scale
    transformed = reward_transformer(scalar, **kwargs)
    floored = np.floor(transformed)  # Lower-bound support integer
    prob = transformed - floored     # Proportion between adjacent integers

    bins = np.zeros(2 * support_size + 1)

    bins[floored + support_size] = 1 - prob
    bins[floored + support_size + 1] = prob

    bins = bins[np.newaxis, ...]  # Shape (1, support_size * 2 + 1)
    return bins





