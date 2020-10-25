"""

"""
import typing
import sys
from datetime import datetime

import numpy as np
from tqdm import trange
import tensorflow as tf

from Coach import Coach
from Arena import Arena
from Experimenter.players import MuZeroPlayer
from MuZero.MuMCTS import MuZeroMCTS
from utils import DotDict
from utils.selfplay_utils import GameHistory, TemperatureScheduler, sample_batch


class MuZeroCoach(Coach):
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, neural_net, args: DotDict) -> None:
        """

        :param game:
        :param neural_net:
        :param args:
        """
        super().__init__(game, args)
        self.observation_encoding = game.Representation.HEURISTIC

        self.neural_net = neural_net
        self.mcts = MuZeroMCTS(self.game, self.neural_net, self.args)
        self.arena_player = MuZeroPlayer(self.game, self.mcts, self.neural_net, DotDict({'name': 'p1'}))

        if self.args.pitting:
            self.opponent_net = self.neural_net.__class__(self.game, neural_net.net_args, self.neural_net.architecture)
            self.opponent_mcts = MuZeroMCTS(self.game, self.opponent_net, self.args)
            self.arena_opponent = MuZeroPlayer(self.game, self.opponent_mcts, self.opponent_net, DotDict({'name': 'p2'}))

        self.temp_schedule = TemperatureScheduler(self.args.temperature_schedule)
        self.update_temperature = self.temp_schedule.build()

        self.logdir = "logs/MuZero/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # TODO specify path in file
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
        actions = history.actions[t:t+k]  # Actions are shifted one step to the right.

        # Targets
        pis = history.probabilities[t:t+k+1]
        vs = history.observed_returns[t:t+k+1]
        rewards = history.rewards[t:t+k+1]

        # Handle truncations > 0 due to terminal states. Treat last state as absorbing state
        a_truncation = k - len(actions)  # Action truncation
        if a_truncation > 0:
            actions += np.random.choice(self.game.getActionSize(), size=a_truncation).tolist()

        t_truncation = (k + 1) - len(pis)  # Target truncation due to terminal state
        if t_truncation > 0:
            pis += [pis[-1]] * t_truncation  # Uniform policy
            rewards += [rewards[-1]] * t_truncation
            vs += [0] * t_truncation

        # One hot encode actions.
        enc_actions = np.zeros([len(actions), self.game.getActionSize()])
        enc_actions[np.arange(len(actions)), actions] = 1

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
        current_player = 1
        z = step = 0

        while not z:  # Boardgames: If loop ends => current player lost
            step += 1

            # Update MCTS visit count temperature according to an episode or weight update schedule.
            temp = self.update_temperature(self.neural_net.steps if self.temp_schedule.args.by_weight_update else step)

            # Construct an observation array (o_1, ..., o_t).
            o_t = self.game.buildObservation(state, current_player, form=self.observation_encoding)
            stacked_observations = history.stackObservations(self.neural_net.net_args.observation_length, o_t)

            # Compute the move probability vector and state value using MCTS for the current state of the environment.
            root_actions = self.game.getLegalMoves(state, current_player)  # We can query the env at a current state.
            pi, v = self.mcts.runMCTS(stacked_observations, legal_moves=root_actions, temp=temp)

            # Take a step in the environment and observe the transition and store necessary statistics.
            action = np.random.choice(len(pi), p=pi)
            state, r, next_player = self.game.getNextState(state, action, current_player)
            history.capture(o_t, action, current_player, pi, r, v)

            # Update state of control
            current_player = next_player
            z = self.game.getGameEnded(state, current_player, close=True)

        # Capture terminal state and compute z_t for each observation == N-step returns for general MDPs
        o_terminal = self.game.buildObservation(state, current_player, form=self.observation_encoding)

        # Terminal reward for board games is -1 or 1. For general games the bootstrap value is 0 (future rewards = 0)
        history.terminate(o_terminal, current_player, (z if self.game.n_players > 1 else 0))
        history.compute_returns(gamma=self.args.gamma, n=(self.args.n_steps if self.game.n_players == 1 else None))

        return history

    def learn(self) -> None:
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it trains the neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        Afterwards the current neural network weights are stored and the loop continues.
        """
        for i in range(1, self.args.numIters + 1):
            print(f'------ITER {i}------')
            # Self-play/ Gather training data.
            iteration_train_examples = list()
            for _ in trange(self.args.numEps, desc="Self Play", file=sys.stdout):
                self.mcts.clear_tree()  # Reset the search tree after every game.
                iteration_train_examples.append(self.executeEpisode())

                if sum(map(len, iteration_train_examples)) > self.args.maxlenOfQueue:
                    iteration_train_examples.pop(0)

            # Store data from previous self-play iterations into the history and print out statistics.
            self.trainExamplesHistory.append(iteration_train_examples)
            GameHistory.print_statistics(self.trainExamplesHistory)

            # Backup history to a file
            self.saveTrainExamples(i - 1)

            # Flatten examples over self-play episodes and sample a training batch.
            complete_history = [subitem for item in self.trainExamplesHistory for subitem in item]

            # Training new network, keeping a copy of the old one
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # Backpropagation
            for _ in trange(self.args.numTrainingSteps, desc="Backpropagation", file=sys.stdout):
                batch = self.sampleBatch(complete_history)

                self.neural_net.monitor.log_batch(batch)
                self.neural_net.train(batch)

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
