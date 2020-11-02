"""

"""
import os
import sys
import typing
from pickle import Pickler, Unpickler, HIGHEST_PROTOCOL
from collections import deque

import numpy as np
from tqdm import trange

from Arena import Arena
from utils import DotDict
from utils.selfplay_utils import GameHistory, TemperatureScheduler
from utils.debugging import RENDER


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, neural_net, args: DotDict, search_engine, player) -> None:
        """

        :param game:
        :param args:
        """
        self.game = game
        self.args = args

        self.trainExamplesHistory = deque(maxlen=self.args.numItersForTrainExamplesHistory)
        self.skipFirstSelfPlay = False  # Can be overridden in loadTrainExamples()

        # Initialize network and search engine
        self.neural_net = neural_net
        self.mcts = search_engine(self.game, self.neural_net, self.args)
        self.arena_player = player(self.game, self.mcts, self.neural_net, DotDict({'name': 'p1'}))

        if self.args.pitting:
            self.opponent_net = self.neural_net.__class__(self.game, neural_net.net_args, neural_net.architecture)
            self.opponent_mcts = search_engine(self.game, self.opponent_net, self.args)
            self.arena_opponent = player(self.game, self.opponent_mcts, self.opponent_net, DotDict({'name': 'p2'}))

        self.temp_schedule = TemperatureScheduler(self.args.temperature_schedule)
        self.update_temperature = self.temp_schedule.build()

    @staticmethod
    def getCheckpointFile(iteration: int) -> str:
        return f'checkpoint_{iteration}.pth.tar'

    def sampleBatch(self, histories: typing.List[GameHistory]) -> typing.List:
        """
        Sample a batch of data from the current replay buffer (with or without prioritization).

        Returns:
            A list of sample statistics of the form (observation, targets, meta_data)
        """
        raise NotImplementedError("Coach.py has no default batch sample procedure.")

    def executeEpisode(self) -> GameHistory:
        """
        This function executes one episode of self-play, starting with player 1.

        It uses a temp=1 if episode_step < tempThreshold, and thereafter uses temp=0.

        Returns:
            history: A data structure containing all observed states and statistics
                     from the perspective of the past current players.
                     The structure is of the form (s_t, a_t, player_t, pi_t, r_t, v_t, z_t)
        """
        history = GameHistory()
        state = self.game.getInitialState()  # Always from perspective of player 1 for boardgames.
        z = step = 0

        while not state.done:  # Boardgames: If loop ends => current player lost
            step += 1

            if RENDER:  # Display visualization of the environment if specified.
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
            z = self.game.getGameEnded(state, close=True)

        # Terminal reward for board games is -1 or 1. For general games the bootstrap value is 0 (future rewards = 0)
        history.terminate(state, (z if self.game.n_players > 1 else 0))
        history.compute_returns(gamma=self.args.gamma, n=(self.args.n_steps if self.game.n_players == 1 else None))

        return history

    def learn(self) -> None:
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in complete_history (which has a maximum length of maxlenofQueue).
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
            complete_history = GameHistory.flatten(self.trainExamplesHistory)

            # Training new network, keeping a copy of the old one
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # Backpropagation
            for _ in trange(self.args.numTrainingSteps, desc="Backpropagation", file=sys.stdout):
                batch = self.sampleBatch(complete_history)

                self.neural_net.train(batch)
                self.neural_net.monitor.log_batch(batch)

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

    def saveTrainExamples(self, iteration: int) -> None:
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
        model_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            print(examples_file)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examples_file, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
