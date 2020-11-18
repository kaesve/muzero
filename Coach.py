"""
Define the base self-play/ data gathering class. This class should work with any MCTS-based neural network learning
algorithm like AlphaZero or MuZero. Self-play, model-fitting, and pitting is performed sequentially on a single-thread
in this default implementation.

Notes:
 - Code adapted from https://github.com/suragnair/alpha-zero-general
 - Base implementation done.
 - Base implementation sufficiently abstracted to accommodate both AlphaZero and MuZero.
 - Documentation 15/11/2020
"""
import os
import sys
import typing
from pickle import Pickler, Unpickler, HIGHEST_PROTOCOL
from collections import deque
from abc import ABC, abstractmethod

import numpy as np
from tqdm import trange

from Experimenter import Arena
from utils import DotDict
from utils.selfplay_utils import GameHistory, TemperatureScheduler
from utils import debugging


class Coach(ABC):
    """
    This class controls the self-play and learning loop. Subclass this abstract class to define implementation
    specific procedures for sampling data for the learning algorithm. See MuZero/MuNeuralNet.py or
    AlphaZero/AlphaNeuralNet.py for examples.
    """

    def __init__(self, game, neural_net, args: DotDict, search_engine, player) -> None:
        """
        Initialize the self-play class with an environment, an agent to train, requisite hyperparameters, a MCTS search
        engine, and an agent-interface.
        :param game: Game Implementation of Game class for environment logic.
        :param neural_net: Some implementation of a neural network class to be trained.
        :param args: DotDict Data structure containing parameters for self-play.
        :param search_engine: Class containing the logic for performing MCTS using the neural_net.
        :param player: Class containing the logic for agent-environment interaction.
        """
        self.game = game
        self.args = args

        # Initialize replay buffer and helper variable
        self.trainExamplesHistory = deque(maxlen=self.args.selfplay_buffer_window)
        self.update_on_checkpoint = False  # Can be overridden in loadTrainExamples()

        # Initialize network and search engine
        self.neural_net = neural_net
        self.mcts = search_engine(self.game, self.neural_net, self.args)
        self.arena_player = player(self.game, None)
        self.arena_player.set_variables(self.neural_net, self.mcts, 'p1')

        # Initialize adversary if specified.
        if self.args.pitting:
            self.opponent_net = self.neural_net.__class__(self.game, neural_net.net_args, neural_net.architecture)
            self.opponent_mcts = search_engine(self.game, self.opponent_net, self.args)
            self.arena_opponent = player(self.game, None)
            self.arena_opponent.set_variables(self.opponent_net, self.opponent_mcts, 'p2')

        # Initialize MCTS visit count exponentiation factor schedule.
        self.temp_schedule = TemperatureScheduler(self.args.temperature_schedule)
        self.update_temperature = self.temp_schedule.build()

    @staticmethod
    def getCheckpointFile(iteration: int) -> str:
        """ Helper function to format model checkpoint filenames """
        return f'checkpoint_{iteration}.pth.tar'

    @abstractmethod
    def sampleBatch(self, histories: typing.List[GameHistory]) -> typing.List:
        """
        Sample a batch of data from the current replay buffer (with or without prioritization).

        This method is left abstract as different algorithm instances may require different data-targets.

        :param histories: List of GameHistory objects. Contains all game-trajectories in the replay-buffer.
        :return: List of training examples.
        """

    def executeEpisode(self) -> GameHistory:
        """
        Perform one episode of self-play for gathering data to train neural networks on.

        The implementation details of the neural networks/ agents, temperature schedule, data storage
        is kept highly transparent on this side of the algorithm. Hence for implementation details
        see the specific implementations of the function calls.

        At every step we record a snapshot of the state into a GameHistory object, this includes the observation,
        MCTS search statistics, performed action, and observed rewards. After the end of the episode, we close the
        GameHistory object and compute internal target values.

        :return: GameHistory Data structure containing all observed states and statistics required for network training.
        """
        history = GameHistory()
        state = self.game.getInitialState()  # Always from perspective of player 1 for boardgames.
        step = 1

        while not state.done and step <= self.args.max_episode_moves:
            if debugging.RENDER:  # Display visualization of the environment if specified.
                self.game.render(state)

            # Update MCTS visit count temperature according to an episode or weight update schedule.
            temp = self.update_temperature(self.neural_net.steps if self.temp_schedule.args.by_weight_update else step)

            # Compute the move probability vector and state value using MCTS for the current state of the environment.
            pi, v = self.mcts.runMCTS(state, history, temp=temp)

            # Take a step in the environment and observe the transition and store necessary statistics.
            state.action = np.random.choice(len(pi), p=pi)
            next_state, r = self.game.getNextState(state, state.action)
            history.capture(state, pi, r, v)

            # Update state of control
            state = next_state
            step += 1

        # Cleanup environment and GameHistory
        self.game.close(state)
        history.terminate()
        history.compute_returns(gamma=self.args.gamma, n=(self.args.n_steps if self.game.n_players == 1 else None))

        return history

    def learn(self) -> None:
        """
        Control the data gathering and weight optimization loop. Perform 'num_selfplay_iterations' iterations
        of self-play to gather data, each of 'num_episodes' episodes. After every self-play iteration, train the
        neural network with the accumulated data. If specified, the previous neural network weights are evaluated
        against the newly fitted neural network weights, the newly fitted weights are then accepted based on some
        specified win/ lose ratio. Neural network weights and the replay buffer are stored after every iteration.
        Note that for highly granular vision based environments, that the replay buffer may grow to large sizes.
        """
        for i in range(1, self.args.num_selfplay_iterations + 1):
            print(f'------ITER {i}------')
            if not self.update_on_checkpoint or i > 1:  # else: go directly to backpropagation

                # Self-play/ Gather training data.
                iteration_train_examples = list()
                for _ in trange(self.args.num_episodes, desc="Self Play", file=sys.stdout):
                    self.mcts.clear_tree()
                    iteration_train_examples.append(self.executeEpisode())

                    if sum(map(len, iteration_train_examples)) > self.args.max_buffer_size:
                        iteration_train_examples.pop(0)

                # Store data from previous self-play iterations into the history.
                self.trainExamplesHistory.append(iteration_train_examples)

            # Print out statistics about the replay buffer, and back-up the data history to a file (can be slow).
            GameHistory.print_statistics(self.trainExamplesHistory)
            self.saveTrainExamples(i - 1)

            # Flatten examples over self-play episodes and sample a training batch.
            complete_history = GameHistory.flatten(self.trainExamplesHistory)

            # Training new network, keeping a copy of the old one
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # Backpropagation
            for _ in trange(self.args.num_gradient_steps, desc="Backpropagation", file=sys.stdout):
                batch = self.sampleBatch(complete_history)

                self.neural_net.train(batch)
                self.neural_net.monitor.log_batch(batch)

            # Pitting
            accept = True
            if self.args.pitting:
                # Load in the old network.
                self.opponent_net.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

                # Perform trials with the new network against the old network.
                arena = Arena(self.game, self.arena_player, self.arena_opponent, self.args.max_trial_moves)
                accept = arena.pitting(self.args, self.neural_net.monitor)

            if accept:
                print('ACCEPTING NEW MODEL')
                self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.args.load_folder_file[-1])
            else:
                print('REJECTING NEW MODEL')
                self.neural_net.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

    def saveTrainExamples(self, iteration: int) -> None:
        """
        Store the current accumulated data to a compressed file using pickle. Note that for highly dimensional
        environments, that the stored files may be considerably large and that storing/ loading the data may
        introduce a significant bottleneck to the runtime of the algorithm.
        :param iteration: int Current iteration of the self-play. Used as indexing value for the data filename.
        """
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f, protocol=HIGHEST_PROTOCOL).dump(self.trainExamplesHistory)

        # Don't hog up storage space and clean up old (never to be used again) data.
        old_checkpoint = os.path.join(folder, self.getCheckpointFile(iteration - 1) + '.examples')
        if os.path.isfile(old_checkpoint):
            os.remove(old_checkpoint)

    def loadTrainExamples(self) -> None:
        """
        Load in a previously generated replay buffer from the path specified in the .json arguments.
        """
        model_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            r = input(f"Data file {examples_file} could not be found. Continue with a fresh buffer? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print(f"Data file {examples_file} found. Read it.")
            with open(examples_file, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
