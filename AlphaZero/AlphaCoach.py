"""
Implements the abstract Coach class for defining the data sampling procedures for AlphaZero neural network training.

Notes:
 - Base implementation done.
 - Documentation 15/11/2020
"""
import typing
from datetime import datetime

import tensorflow as tf

from Coach import Coach
from AlphaZero.AlphaMCTS import MCTS
from Agents import DefaultAlphaZeroPlayer
from utils.selfplay_utils import GameHistory, sample_batch
from utils import DotDict


class AlphaZeroCoach(Coach):
    """
    Implement base Coach class to define proper data-batch sampling procedures and logging objects.
    """

    def __init__(self, game, neural_net, args: DotDict, run_name: typing.Optional[str] = None) -> None:
        """
        Initialize the class for self-play. This inherited method initializes tensorboard logging.

        The super class is initialized with the proper search engine and agent-interface. (MCTS, AlphaZeroPlayer)

        :param game: Game Implementation of Game class for environment logic.
        :param neural_net: AlphaNeuralNet Implementation of AlphaNeuralNet class for inference.
        :param args: DotDict Data structure containing parameters for self-play.
        :param run_name: str Optionally provide a run-name for the TensorBoard log-files. Default is current datetime.
        """
        super().__init__(game, neural_net, args, MCTS, DefaultAlphaZeroPlayer)

        # Initialize tensorboard logging.
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.log_dir = f"out/logs/AlphaZero/{self.neural_net.architecture}/" + run_name
        self.file_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        self.file_writer.set_as_default()

    def sampleBatch(self, histories: typing.List[GameHistory]) -> typing.List:
        """
        Construct a batch of data-targets for gradient optimization of the AlphaZero neural network.

        The procedure samples a list of game and inside-game coordinates of length 'batch_size'. This is done either
        uniformly or with prioritized sampling. Using this list of coordinates, we sample the according games, and
        the according points of times within the game to generate neural network inputs, targets, and sample weights.

        The targets for the neural network consist of MCTS move probability vectors and TD/ Monte-Carlo returns.

        :param histories: List of GameHistory objects. Contains all game-trajectories in the replay-buffer.
        :return: List of training examples: (observations, (move-probabilities, TD/ MC-returns), sample_weights)
        """
        # Generate coordinates within the replay buffer to sample from. Also generate the loss scale of said samples.
        sample_coordinates, sample_weight = sample_batch(
            list_of_histories=histories, n=self.neural_net.net_args.batch_size, prioritize=self.args.prioritize,
            alpha=self.args.prioritize_alpha, beta=self.args.prioritize_beta)

        # Collect training examples for AlphaZero: (o_t, (pi_t, v_t), w_t)
        examples = [(
            histories[h_i].stackObservations(self.neural_net.net_args.observation_length, t=i),
            (histories[h_i].probabilities[i], histories[h_i].observed_returns[i]),
            loss_scale
        )
            for (h_i, i), loss_scale in zip(sample_coordinates, sample_weight)
        ]

        return examples
