"""
TODO Revamp to work with single player games. Prioritized sampling.
"""
import sys
import typing
from datetime import datetime

import numpy as np
from tqdm import trange
import tensorflow as tf

from Coach import Coach
from Arena import Arena
from AlphaZero.AlphaMCTS import MCTS
from Experimenter.players import AlphaZeroPlayer
from utils.selfplay_utils import GameHistory, TemperatureScheduler
from utils import DotDict


class AlphaZeroCoach(Coach):
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, neural_net, args: DotDict) -> None:
        super().__init__(game, args)
        self.observation_encoding = game.Representation.CANONICAL
        self.skipFirstSelfPlay = False  # Can be overridden in loadTrainExamples()

        self.neural_net = neural_net
        self.mcts = MCTS(self.game, self.neural_net, self.args)
        self.arena_player = AlphaZeroPlayer(self.game, self.mcts, self.neural_net, DotDict({'name': 'p1'}))

        if self.args.pitting:
            self.opponent_net = self.neural_net.__class__(self.game, neural_net.net_args, neural_net.architecture)
            self.opponent_mcts = MCTS(self.game, self.opponent_net, self.args)
            self.arena_opponent = AlphaZeroPlayer(self.game, self.opponent_mcts, self.opponent_net, DotDict({'name': 'p2'}))

        self.temp_schedule = TemperatureScheduler(self.args.temperature_schedule)
        self.update_temperature = self.temp_schedule.build()

        self.logdir = "logs/AlphaZero/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # TODO specify path in file
        self.file_writer = tf.summary.create_file_writer(self.logdir + "/metrics")
        self.file_writer.set_as_default()

    @staticmethod
    def getCheckpointFile(iteration: int) -> str:
        return f'checkpoint_{iteration}.pth.tar'

    def flattenTrainingExamples(self) -> typing.List[typing.Tuple[np.ndarray, np.ndarray, float]]:
        """
        Flatten histories to list of training examples over all state symmetries.

        Returned list is of the form [[o_1, pi_1, v_1], [o_2, pi_2, v_2], ...]
        """
        flat_history = [subitem for item in self.trainExamplesHistory for subitem in item]

        examples = list()
        for history in flat_history:
            examples += [[*symmetry, history.observed_returns[t]]
                         for t in range(len(history)) for symmetry in history.symmetries[t]]
        # TODO: Priority sampling? Check how data sampling and backprop was done for AlphaZero
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

    def learn(self) -> None:
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            print(f'------ITER {i}------')
            if not self.skipFirstSelfPlay or i > 1:  # else: go directly to backpropagation

                # Self-play/ Gather training data.
                iteration_train_examples = list()
                for _ in trange(self.args.numEps, desc="Self Play", file=sys.stdout):
                    self.mcts.clear_tree()
                    iteration_train_examples.append(self.executeEpisode())

                    if sum(map(len, iteration_train_examples)) > self.args.maxlenOfQueue:
                        iteration_train_examples.pop(0)

                # Store data from previous self-play iterations into the history.
                self.trainExamplesHistory.append(iteration_train_examples)

            # Print out statistics about the replay buffer.
            GameHistory.print_statistics(self.trainExamplesHistory)

            # Backup history to a file
            self.saveTrainExamples(i - 1)

            # Flatten examples over self-play episodes and sample a training batch.
            train_examples = self.flattenTrainingExamples()

            # Training new network, keeping a copy of the old one
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # Backpropagation
            print("Training the neural network weights.")
            self.neural_net.train(train_examples, self.args.numTrainingSteps)

            # Pitting
            accept = True
            if self.args.pitting:
                # Load in the old network.
                self.opponent_net.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

                # Perform trials with the new network against the old network.
                arena = Arena(self.game, self.arena_player, self.arena_opponent)
                accept = arena.pitting(self.args, self.neural_net.monitor)

            if accept:
                print('ACCEPTING NEW MODEL')
                self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.args.load_folder_file[-1])
            else:
                print('REJECTING NEW MODEL')
                self.neural_net.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
