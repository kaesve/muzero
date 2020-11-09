"""

"""
import typing
from datetime import datetime

import numpy as np
import tensorflow as tf

from Coach import Coach
from Experimenter.players import MuZeroPlayer
from MuZero.MuMCTS import MuZeroMCTS
from utils import DotDict
from utils.selfplay_utils import GameHistory, sample_batch


class MuZeroCoach(Coach):
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, neural_net, args: DotDict, run_name: typing.Optional[str] = None) -> None:
        """

        :param game:
        :param neural_net:
        :param args:
        """
        super().__init__(game, neural_net, args, MuZeroMCTS, MuZeroPlayer)

        if run_name == None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.logdir = f"out/logs/MuZero/{self.neural_net.architecture}/" + run_name
        self.file_writer = tf.summary.create_file_writer(self.logdir + "/metrics")
        self.file_writer.set_as_default()

    def buildHypotheticalSteps(self, history: GameHistory, t: int, k: int) -> \
            typing.Tuple[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """

        :param history:
        :param t:
        :param k:
        :return:
        """
        # One hot encode actions. Keep truncated actions empty (zeros).
        actions = history.actions[t:t+k]  # Actions are shifted one step to the right.
        a_truncation = k - len(actions)
        if a_truncation > 0:
            actions += np.random.randint(self.game.getActionSize(), size=a_truncation).tolist()

        enc_actions = np.zeros([k, self.game.getActionSize()])
        enc_actions[np.arange(len(actions)), actions] = 1

        # Targets
        pis = history.probabilities[t:t+k+1]
        vs = history.observed_returns[t:t+k+1]
        rewards = history.rewards[t:t+k+1]

        # Handle truncations > 0 due to terminal states. Treat last state as absorbing state
        t_truncation = (k + 1) - len(pis)  # Target truncation due to terminal state
        if t_truncation > 0:
            pis += [pis[-1]] * t_truncation  # Uniform policy
            rewards += [rewards[-1]] * t_truncation
            vs += [0] * t_truncation

        return enc_actions, (np.asarray(vs), np.asarray(rewards), np.asarray(pis))  # (Actions, Targets)

    def sampleBatch(self, histories: typing.List[GameHistory]) -> typing.List:
        """

        :param histories:
        :return:
        """
        # Generate coordinates within the replay buffer to sample from. Also generate the loss scale of said samples.
        sample_coordinates, sample_weight = sample_batch(
            list_of_histories=histories, n=self.neural_net.net_args.batch_size, prioritize=self.args.prioritize,
            alpha=self.args.prioritize_alpha, beta=self.args.prioritize_beta)

        # Construct training examples for MuZero of the form (input, action, (targets), loss_scalar)
        examples = [(
            histories[h_i].stackObservations(self.neural_net.net_args.observation_length, t=i),
            *self.buildHypotheticalSteps(histories[h_i], i, k=self.args.K),
            loss_scale
        )
            for (h_i, i), loss_scale in zip(sample_coordinates, sample_weight)
        ]

        return examples
