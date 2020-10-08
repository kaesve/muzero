import unittest

import numpy as np

from utils.loss_utils import scalar_to_support, support_to_scalar, atari_reward_transform, \
    inverse_atari_reward_transform
from utils.selfplay_utils import GameHistory


class TestStaticFunctions(unittest.TestCase):

    def test_reward_functions(self):  # TODO atari reward transforms
        pass

    def test_reward_transformations(self):
        bins = 300  # Ensure that bins is large enough to support 'high'.
        n = 10
        high = 1e4

        # Generate some random (large) values
        scalars = np.random.randn(n) * high

        # Cast scalars to support points of a categorical distribution.
        support = scalar_to_support(scalars, bins)

        # Ensure correct dimensionality
        self.assertEquals(support.shape, (n, bins * 2 + 1))

        # Cast support points back to scalars.
        inverted = support_to_scalar(support, bins)

        # Ensure correct dimensionality
        self.assertEquals(inverted.shape, scalars.shape)

        # Scalar to support and back to scalars should be equal.
        np.testing.assert_array_almost_equal(scalars, inverted)

    def test_n_step_return_estimation(self):  # TODO n-step return estimation
        h = GameHistory()


if __name__ == '__main__':
    unittest.main()
