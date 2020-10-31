"""
"""
import typing
from datetime import datetime

import numpy as np
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
        self.observation_encoding = game.Representation.CANONICAL

        self.temp_schedule = TemperatureScheduler(self.args.temperature_schedule)
        self.update_temperature = self.temp_schedule.build()

        self.logdir = f"logs/AlphaZero/{self.neural_net.architecture}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
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

        # Randomly select target data from available symmetries
        examples = list()
        for (h_i, i), loss_scale in zip(sample_coordinates, sample_weight):
            symmetries = histories[h_i].symmetries[i]
            choice = symmetries[np.random.randint(len(symmetries))]
            # Collect (o_t, (pi_t, v_t), w_t)
            examples.append((choice[0], (choice[1], histories[h_i].observed_returns[i]), loss_scale))

        return examples

    def executeEpisode(self) -> GameHistory:
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temp=1 if step < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            train_examples: a list of examples of the form (canonical_state, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        history = GameHistory()
        state = self.game.getInitialState()
        current_player = 1
        z = step = 0

        while not z:
            step += 1
            # Update MCTS visit count temperature according to an episode or weight update schedule.
            temp = self.update_temperature(self.neural_net.steps if self.temp_schedule.args.by_weight_update else step)

            # Compute the move probability vector and state value using MCTS for the current state of the environment.
            pi, v = self.mcts.runMCTS(state, temp=temp)

            # Take a step in the environment and observe the transition
            action = np.random.choice(len(pi), p=pi)
            state, r, next_player = self.game.getNextState(state, action, current_player)

            # Store necessary statistics. Also construct state representation := neural network input
            o_t = self.game.buildObservation(state, current_player, form=self.observation_encoding)
            history.capture(o_t, action, current_player, pi, r, v)
            history.find_symmetries(self.game)

            # Update state of control
            current_player = next_player
            z = self.game.getGameEnded(state, current_player, close=True)

        # Capture terminal state and compute z_t for each observation == N-step returns for general MDPs
        o_terminal = self.game.buildObservation(state, current_player, form=self.observation_encoding)

        # Terminal reward for board games is -1 or 1. For general games the bootstrap value is 0 (future rewards = 0)
        history.terminate(o_terminal, current_player, (z if self.game.n_players > 1 else 0))
        history.compute_returns(gamma=self.args.gamma, n=(self.args.n_steps if self.game.n_players == 1 else None))
        history.find_symmetries(self.game)

        return history
