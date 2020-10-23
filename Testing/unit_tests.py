import typing
import unittest
import os
import time

import numpy as np

from Games.gym.GymGame import GymGame
from Games.gym.MuZeroModel.NNet import NNetWrapper as GymNet
from Games.hex.HexGame import HexGame
from Games.hex.MuZeroModel.NNet import NNetWrapper as HexNet

from MuZero.MuMCTS import MuZeroMCTS

from utils import DotDict
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
        n = 10  # Number of samples to draw
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

        # Test bin creation explicitly against manually calculated example.
        scalars = [-2.5, -0.75, 0.2, 1.38, 2.99]
        expected = [
            [0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0.75, 0.25, 0, 0, 0],
            [0, 0, 0, 0.8, 0.2, 0, 0],
            [0, 0, 0, 0, 0.62, 0.38, 0],
            [0, 0, 0, 0, 0, 0.01, 0.99]
        ]
        bins = scalar_to_support(scalars, 3, reward_transformer=lambda x: x)
        np.testing.assert_array_almost_equal(expected, bins)

    def test_n_step_return_estimation_MDP(self):
        horizon = 3  # n-step lookahead for computing z_t
        gamma = 1 / 2  # discount factor for future rewards and bootstrap

        # Experiment settings
        search_results = [5, 5, 5, 5, 5]  # MCTS v_t index +k
        dummy_rewards = [0, 1, 2, 3, 4]  # u_t+1 index +k
        z = 0  # Final return provided by the env.

        # Desired output: Correct z_t index +k (calculated manually)
        n_step = [1 + 5 / 8, 3 + 3 / 8, 4 + 1 / 2, 5.0, 4.0, 0]

        # Fill the GameHistory with the required data.
        h = GameHistory()
        for r, v in zip(dummy_rewards, search_results):
            h.capture(np.array([0]), -1, 1, np.array([0]), r, v)
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

        np.testing.assert_array_almost_equal(stacked_5[..., :-8], 0)  # Should be only 0s
        np.testing.assert_array_almost_equal(stacked_5[..., -8:], dummy_observations[0])  # Should be the first o_t

        # Check whether observation concatenation works correctly
        stacked = h.stackObservations(2, dummy_observations[1])
        expected = np.concatenate(dummy_observations[:2], axis=-1)
        np.testing.assert_array_almost_equal(stacked, expected)

        # Fill the buffer
        all([h.capture(x, -1, 1, np.array([]), 0, 0) for x in dummy_observations[1:]])

        # Check whether time indexing works correctly
        stacked_1to5 = h.stackObservations(4, t=4)  # 1-4 --> t is inclusive
        stacked_last4 = h.stackObservations(4, t=9)  # 6-9
        expected_1to5 = np.concatenate(dummy_observations[1:5], axis=-1)  # t in {1, 2, 3, 4}
        expected_last4 = np.concatenate(dummy_observations[-4:], axis=-1)  # t in {6, 7, 8, 9}

        np.testing.assert_array_almost_equal(stacked_1to5, expected_1to5)
        np.testing.assert_array_almost_equal(stacked_last4, expected_last4)

        # Check if clearing works correctly
        h.refresh()
        self.assertEqual(len(h), 0)


class TestHexMuZero(unittest.TestCase):
    """
    Unit testing class to test whether the search engine exhibit well defined behaviour.
    This includes scenarios where either the model or inputs are faulty (empty observations,
    constant predictions, nans/ inf in observations).
    """
    hex_board_size: int = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Setup required for unit tests.
        print("Unit testing CWD:", os.getcwd())
        self.config = DotDict.from_json("../Experimenter/MuZeroConfigs/boardgames.json")
        self.g = HexGame(self.hex_board_size)
        self.net = HexNet(self.g, self.config.net_args)
        self.mcts = MuZeroMCTS(self.g, self.net, self.config.args)

    def test_empty_input(self):
        """
        Tests the following scenarios:
         - Assert that observation tensors with only zeros are encoded to finite values (can be zero)
         - Assert that latent state tensors with only zeros are transitioned to finite values (can be zero)
        """
        # Build the environment for an observation.
        s = self.g.getInitialState()
        o_t = self.g.buildObservation(s, player=1, form=self.g.Representation.HEURISTIC)
        h = GameHistory()

        # Build empty observations
        h.capture(o_t, -1, 1, np.array([]), 0, 0)
        stacked = h.stackObservations(self.net.net_args.observation_length, o_t)
        zeros_like = np.zeros_like(stacked)

        # Check if nans are produced
        latent, _, _ = self.net.initial_inference(zeros_like)
        self.assertTrue(np.isfinite(latent).all())

        # Exhaustively ensure that all possible dynamics function inputs lead to finite values.
        latent_forwards = [self.net.recurrent_inference(latent, action)[1] for action in range(self.g.getActionSize())]
        self.assertTrue(np.isfinite(np.array(latent_forwards)).all())

    def test_search_recursion_error(self):
        """
        The main phenomenon this test attempts to find is:
        Let s be the current latent state, s = [0, 0, 0], along with action a = 1.
        If we fetch the next latent state with (s, a) we do not want to get, s' == s = [0, 0, 0].
        s' is a new state, although it is present in the transition table due to being identical to s.
        if action a = 1 is chosen again by UCB, then this could result in infinite recursion.

        Tests the following scenarios:
         - Assert that MuMCTS does not result in a recursion error when called with the same
           input multiple times without clearing the tree.
         - Assert that MuMCTS does not result in a recursion error when inputs are either zero
           or random.
         - Assert that MuMCTS does not result in a recursion error when only one root action is legal.
        """
        rep = 30  # Repetition factor --> should be high.

        # Build the environment for an observation.
        s = self.g.getInitialState()
        o_t = self.g.buildObservation(s, player=1, form=self.g.Representation.HEURISTIC)
        h = GameHistory()

        # Build empty and random observations tensors
        h.capture(o_t, -1, 1, np.array([]), 0, 0)
        stacked = h.stackObservations(self.net.net_args.observation_length, o_t)
        zeros_like = np.zeros_like(stacked)
        random_like = np.random.rand(*zeros_like.shape)

        # Build root state legal action masks
        legals = np.ones(self.g.getActionSize())
        same = np.zeros_like(legals)
        same[0] = 1  # Can only do one move

        # Execute multiple MCTS runs that will result in recurring tree paths.
        for _ in range(rep):
            self.mcts.runMCTS(zeros_like, legals)  # Empty observations ALL moves at the root
        self.mcts.clear_tree()

        for _ in range(rep):
            self.mcts.runMCTS(zeros_like, same)  # Empty observations ONE move at the root
        self.mcts.clear_tree()

        for _ in range(rep):
            self.mcts.runMCTS(random_like, legals)  # Empty observations ALL moves at the root
        self.mcts.clear_tree()

        for _ in range(rep):
            self.mcts.runMCTS(random_like, same)  # Empty observations ONE move at the root
        self.mcts.clear_tree()

    def test_search_border_cases_latent_state(self):
        """
        Tests the following scenarios:
        - Assert that observation tensors with only infinities or nans result in finite tensors (zeros).
          Testing this phenomenon ensures that bad input is not propagated for more than one step.
          Note that one forward step using bad inputs can already lead to a recursion error in MuMCTS.
          see test_search_recursion_error
       """
        # Build the environment for an observation.
        s = self.g.getInitialState()
        o_t = self.g.buildObservation(s, player=1, form=self.g.Representation.HEURISTIC)
        h = GameHistory()

        # Build empty observations
        h.capture(o_t, -1, 1, np.array([]), 0, 0)
        stacked = h.stackObservations(self.net.net_args.observation_length, o_t)
        nans_like = np.zeros_like(stacked)
        inf_like = np.zeros_like(stacked)

        nans_like[nans_like == 0] = np.nan
        inf_like[inf_like == 0] = np.inf

        # Check if nans are produced
        nan_latent, _, _ = self.net.initial_inference(nans_like)
        inf_latent, _, _ = self.net.initial_inference(inf_like)

        self.assertTrue(np.isfinite(nan_latent).all())
        self.assertTrue(np.isfinite(inf_latent).all())

        nan_latent[nan_latent == 0] = np.nan
        inf_latent[inf_latent == 0] = np.inf

        # Exhaustively ensure that all possible dynamics function inputs lead to finite values.
        nan_latent_forwards = [self.net.recurrent_inference(nan_latent, action)[1] for action in range(self.g.getActionSize())]
        inf_latent_forwards = [self.net.recurrent_inference(inf_latent, action)[1] for action in range(self.g.getActionSize())]

        self.assertTrue(np.isfinite(np.array(nan_latent_forwards)).all())
        self.assertTrue(np.isfinite(np.array(inf_latent_forwards)).all())

    def test_ill_conditioned_model(self):
        """
        Execute all unit tests of this class using a model with badly conditioned weights.
        i.e., large weight magnitudes or only zeros.
        """

        class DumbModel(HexNet):

            def initial_inference(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float]:
                s, pi, v = super().initial_inference(observations)
                return np.zeros_like(s), np.random.uniform(size=len(pi)), 0

            def recurrent_inference(self, latent_state: np.ndarray, action: int) -> typing.Tuple[float, np.ndarray]:
                r, s, pi, v = super().recurrent_inference(latent_state, action)
                return 0, np.zeros_like(latent_state), np.random.uniform(size=len(pi)), 0

        memory_net = self.net
        memory_search = self.mcts

        # Swap class variables
        self.net = DumbModel(self.g, self.config.net_args)
        self.mcts = MuZeroMCTS(self.g, self.net, self.config.args)

        self.test_search_recursion_error()

        # Undo class variables swap
        self.net = memory_net
        self.mcts = memory_search

    def test_combined_model(self):
        # The prediction and dynamics model can be combined into one computation graph.
        # This should be faster than calling the models separately. This test makes
        # sure that the output is still the same, and also shows the time difference.

        batch = 128
        dim = self.g.getDimensions()

        latent_planes = np.random.uniform(size=(batch, dim[0], dim[1]))
        actions = np.floor(np.random.uniform(size=batch) * dim[0] * dim[1])
        actions = actions.astype(int)

        recurrent_inputs = list(zip(latent_planes, actions))

        # This line is just for warm-up, otherwise the timing is unfair.
        combined_results = [self.net.recurrent_inference(latent, a) for latent, a in recurrent_inputs]

        t0 = time.time()
        combined_results = [self.net.recurrent_inference(latent, a) for latent, a in recurrent_inputs]
        t1 = time.time()
        combined_time = t1 - t0

        dynamics_results = [self.net.forward(latent, a) for latent, a in recurrent_inputs]
        predict_results = [self.net.predict(dyn[1]) for dyn in dynamics_results]

        t0 = time.time()
        dynamics_results = [self.net.forward(latent, a) for latent, a in recurrent_inputs]
        predict_results = [self.net.predict(dyn[1]) for dyn in dynamics_results]
        t1 = time.time()
        separate_time = t1 - t0

        print(f"Combined: {combined_time}. Separate: {separate_time}")

        # unzip results
        combined_results = list(zip(*combined_results))
        dynamics_results = list(zip(*dynamics_results))
        predict_results = list(zip(*predict_results))

        np.testing.assert_array_almost_equal(combined_results[0], dynamics_results[0])
        np.testing.assert_array_almost_equal(combined_results[1], dynamics_results[1])
        np.testing.assert_array_almost_equal(combined_results[2], predict_results[0])
        np.testing.assert_array_almost_equal(combined_results[3], predict_results[1])


class TestTreeSearch(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Setup required for unit tests.
        print("Unit testing CWD:", os.getcwd())
        self.config = DotDict.from_json("../Experimenter/MuZeroConfigs/singleplayergames.json")
        self.g = GymGame('CartPole-v1')
        self.net = GymNet(self.g, self.config.net_args)
        self.mcts = MuZeroMCTS(self.g, self.net, self.config.args)

    def test_tree_search(self) -> None:
        """
        This is a behaviour example of MuZero's MCTS procedure that was worked out analytically.
        Model with fixed prior [3/4 - e, 1/4 + e] where e is a small epsilon term to break ties.
        If action 0 is take reward 0 is given, if action 1 is taken, reward 1 is given.
        The Tree MinMax scaling should adjust to the [0, 1] bounds autonomously.

        The proposed model creates a motif in the tree search of the following action trajectories:
         - (a_0), (a_0, a_0), (a_1,)
        i.e., at some node at first a_0 will be chosen two times, afterwards a_1 will be chosen.
        This pattern reoccurs, however, in the recursive fashion of MCTS.

        Hence, for 4 MCTS simulations (ignoring the root), we get the following path:
         - (0), (0, 0), (1,), (1, 0)
        pi, v = MuMCTS should then return pi = [1/2, 1/2] and v = 1/4
        For 8 MCTS simulations (ignoring the root), we should get:
         - (0), (0, 0), (1,), (1, 0), (1, 0, 0), (0, 0, 0), (0, 1), (0, 1, 0)
        pi, v = MuMCTS should then return pi = [5/8, 3/8] and v = 2/8 = 1/4
        """
        class DumbModel(GymNet):
            count: int = 0

            def initial_inference(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float]:
                s, pi, v = super().initial_inference(observations)
                self.count += 1
                return np.ones_like(s) * self.count, np.array([6/8 - 1e-8, 2/8 + 1e-8]), 0

            def recurrent_inference(self, latent_state: np.ndarray, action: int) -> typing.Tuple[float, np.ndarray]:
                r, s, pi, v = super().recurrent_inference(latent_state, action)
                self.count += 1
                return 0, np.ones_like(latent_state) * self.count, np.array([6/8 - 1e-8, 2/8 + 1e-8]), action

        memory_net = self.net
        memory_search = self.mcts

        # Swap class variables
        self.net = DumbModel(self.g, self.config.net_args)
        self.mcts = MuZeroMCTS(self.g, self.net, self.config.args)

        # No discounting and no exploration to ensure deterministic behaviour.
        self.config.args.gamma = 1
        self.config.args.exploration_fraction = 0

        # Experiment 1
        self.config.args.numMCTSSims = 4
        pi_1, v_1 = self.mcts.runMCTS(np.zeros(4), np.ones(2))
        np.testing.assert_array_almost_equal(pi_1, [1/2, 1/2])
        np.testing.assert_almost_equal(v_1, 1/4)
        self.mcts.clear_tree()

        # Experiment 2
        self.config.args.numMCTSSims = 8
        pi_2, v_2 = self.mcts.runMCTS(np.zeros(4), np.ones(2))
        np.testing.assert_array_almost_equal(pi_2, [5/8, 3/8])
        np.testing.assert_almost_equal(v_2, 1/4)
        self.mcts.clear_tree()

        # Undo class variables swap
        self.net = memory_net
        self.mcts = memory_search


class TestSelfPlay(unittest.TestCase):

    def test_history_building(self):
        pass


if __name__ == '__main__':
    unittest.main()
