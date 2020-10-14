import unittest

import numpy as np

from utils.loss_utils import scalar_to_support, support_to_scalar, atari_reward_transform, \
    inverse_atari_reward_transform
from utils.selfplay_utils import GameHistory


class TestStaticFunctions(unittest.TestCase):

    def test_reward_scale_transformation(self):
        var_eps = 0.01  # Paper uses 0.001, but correct function implementation is easier tested with 0.01
        positive_rewards = np.arange(0, 101)
        negative_rewards = np.arange(-100, 1)

        f_positive = atari_reward_transform(positive_rewards, var_eps=var_eps)
        f_negative = atari_reward_transform(negative_rewards, var_eps=var_eps)

        # Assert symmetry
        np.testing.assert_array_almost_equal(f_positive, -f_negative[::-1])

        # Assert correct mapping for var_eps = 0.001. -> f(100) \equiv sqrt(101)
        self.assertAlmostEqual(var_eps, 0.01)
        self.assertAlmostEqual(f_positive[-1], np.sqrt(positive_rewards[-1] + 1))
        self.assertAlmostEqual(f_positive[0], 0)

        inv_positive = inverse_atari_reward_transform(f_positive, var_eps=var_eps)
        inv_negative = inverse_atari_reward_transform(f_negative, var_eps=var_eps)

        # Assert inverse working correctly.
        np.testing.assert_array_almost_equal(inv_positive, positive_rewards)
        np.testing.assert_array_almost_equal(inv_negative, negative_rewards)

    def test_reward_distribution_transformation(self):
        bins = 300  # Ensure that bins is large enough to support 'high'.
        n = 10      # Number of samples to draw
        high = 1e3  # Factor to scale the randomly generated rewards

        # Generate some random (large) values
        scalars = np.random.randn(n) * high

        # Cast scalars to support points of a categorical distribution.
        support = scalar_to_support(scalars, bins)

        # Ensure correct dimensionality
        self.assertEqual(support.shape, (n, bins * 2 + 1))

        # Cast support points back to scalars.
        inverted = support_to_scalar(support, bins)

        # Ensure correct dimensionality
        self.assertEqual(inverted.shape, scalars.shape)

        # Scalar to support and back to scalars should be equal.
        np.testing.assert_array_almost_equal(scalars, inverted)

    def test_n_step_return_estimation_MDP(self):
        horizon = 3    # n-step lookahead for computing z_t
        gamma = 1 / 2  # discount factor for future rewards and bootstrap

        # Experiment settings
        search_results = [5, 5, 5, 5, 5]  # MCTS v_t index +k
        dummy_rewards = [0, 1, 2, 3, 4]   # u_t+1 index +k
        z = 0                             # Final return provided by the env.

        # Desired output: Correct z_t index +k (calculated manually)
        n_step = [1 + 5/8, 3 + 3/8, 4 + 1/2, 5.0, 4.0, 0]

        # Fill the GameHistory with the required data.
        h = GameHistory()
        for r, v in zip(dummy_rewards, search_results):
            h.capture(np.array([]), -1, 1, np.array([]), r, v)
        h.terminate(np.array([]), 1, z)

        # Check if algorithm computes z_t's correctly
        h.compute_returns(gamma, horizon)
        np.testing.assert_array_almost_equal(h.observed_returns[:-1], n_step[:-1])

    def test_observation_stacking(self):
        # random normal variables in the form (x, y, c)
        shape = (3, 3, 8)
        dummy_observations = [np.random.randn(np.prod(shape)).reshape(shape) for _ in range(10)]

        h = GameHistory()
        h.capture(dummy_observations[0], -1, 1, np.array([]), 0, 0)

        # Ensure correct shapes and content
        stacked_0 = h.stackObservations(0)
        stacked_1 = h.stackObservations(1)
        stacked_5 = h.stackObservations(5)

        np.testing.assert_array_equal(stacked_0.shape, shape)  # Shape didn't change
        np.testing.assert_array_equal(stacked_1.shape, shape)  # Shape didn't change
        np.testing.assert_array_equal(stacked_5.shape, np.array(shape) * np.array([1, 1, 5]))  # Channels * 5

        np.testing.assert_array_almost_equal(stacked_0, dummy_observations[0])  # Should be the same
        np.testing.assert_array_almost_equal(stacked_1, dummy_observations[0])  # Should be the same

        np.testing.assert_array_almost_equal(stacked_5[..., :-8], 0)            # Should be only 0s
        np.testing.assert_array_almost_equal(stacked_5[..., -8:], dummy_observations[0])  # Should be the first o_t

        # Check whether observation concatenation works correctly
        stacked = h.stackObservations(2, dummy_observations[1])
        expected = np.concatenate(dummy_observations[:2], axis=-1)
        np.testing.assert_array_almost_equal(stacked, expected)

        # Fill the buffer
        all([h.capture(x, -1, 1, np.array([]), 0, 0) for x in dummy_observations[1:]])

        # Check whether time indexing works correctly
        stacked_1to5 = h.stackObservations(4, t=4)   # 1-4 --> t is inclusive
        stacked_last4 = h.stackObservations(4, t=9)  # 6-9
        expected_1to5 = np.concatenate(dummy_observations[1:5], axis=-1)   # t in {1, 2, 3, 4}
        expected_last4 = np.concatenate(dummy_observations[-4:], axis=-1)  # t in {6, 7, 8, 9}

        np.testing.assert_array_almost_equal(stacked_1to5, expected_1to5)
        np.testing.assert_array_almost_equal(stacked_last4, expected_last4)

        # Check if clearing works correctly
        h.refresh()
        self.assertEqual(len(h), 0)


class TestMuZero(unittest.TestCase):

    def test_search_empty_input(self):
        pass

    def test_search_recurring_history(self):
        pass

    def test_search_border_cases_latent_state(self):
        # Only zeros in latent state, nans, infinities, etc.
        pass


class TestSelfPlay(unittest.TestCase):

    def test_history_building(self):
        pass


if __name__ == '__main__':
    unittest.main()
