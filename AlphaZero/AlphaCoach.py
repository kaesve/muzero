"""
"""
import typing
from datetime import datetime

import tensorflow as tf

from Coach import Coach
from AlphaZero.AlphaMCTS import MCTS
from Experimenter.players import AlphaZeroPlayer
from utils.selfplay_utils import GameHistory, TemperatureScheduler, sample_batch
from utils import DotDict


class AlphaZeroCoach(Coach):
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, neural_net, args: DotDict) -> None:
        super().__init__(game, neural_net, args, MCTS, AlphaZeroPlayer)
        self.temp_schedule = TemperatureScheduler(self.args.temperature_schedule)
        self.update_temperature = self.temp_schedule.build()

        self.logdir = f"out/logs/AlphaZero/{self.neural_net.architecture}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.create_file_writer(self.logdir + "/metrics")
        self.file_writer.set_as_default()

    @staticmethod
    def getCheckpointFile(iteration: int) -> str:
        return f'checkpoint_{iteration}.pth.tar'

    def sampleBatch(self, histories: typing.List[GameHistory]) -> typing.List:
        """

        :param histories:
        :return:
        """
        # Generate coordinates within the replay buffer to sample from. Also generate the loss scale of said samples.
        sample_coordinates, sample_weight = sample_batch(
            list_of_histories=histories, n=self.neural_net.net_args.batch_size, prioritize=self.args.prioritize,
            alpha=self.args.prioritize_alpha, beta=self.args.prioritize_beta)

        # Collect (o_t, (pi_t, v_t), w_t)
        examples = [(
            histories[h_i].stackObservations(self.neural_net.net_args.observation_length, t=i),
            (histories[h_i].probabilities[i], histories[h_i].observed_returns[i]),
            loss_scale
        )
            for (h_i, i), loss_scale in zip(sample_coordinates, sample_weight)
        ]

        return examples
