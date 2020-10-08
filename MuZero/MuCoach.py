"""

"""
import typing
import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler

import numpy as np

from MuZero.MuNeuralNet import MuZeroNeuralNet
from MuZero.MuMCTS import MuZeroMCTS
from utils import Bar, AverageMeter, DotDict
from utils.selfplay_utils import GameHistory, sample_batch


class MuZeroCoach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, neural_net: MuZeroNeuralNet, args: DotDict) -> None:
        """

        :param game:
        :param neural_net:
        :param args:
        """
        self.game = game
        self.neural_net = neural_net
        self.args = args
        self.mcts = MuZeroMCTS(self.game, self.neural_net, self.args)
        self.trainExamplesHistory = deque(maxlen=self.args.numItersForTrainExamplesHistory)

    @staticmethod
    def getCheckpointFile(iteration: int) -> str:
        return f'checkpoint_{iteration}.pth.tar'

    def buildHypotheticalSteps(self, history: GameHistory, t: int, k: int) -> \
            typing.Tuple[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """

        :param history:
        :param t:
        :param k:
        :return:
        """
        start = t
        end = t + k
        actions = history.actions[start:end]  # Actions are shifted one step to the right.

        # Targets
        pis = history.probabilities[start:end+1]
        vs = history.observed_returns[start:end + 1]
        rewards = history.rewards[start:end+1]

        # Handle truncations > 0 due to terminal states. Treat last state as absorbing state
        a_truncation = k - len(actions)  # Action truncation
        if a_truncation > 0:
            actions += [actions[-1]] * a_truncation

        t_truncation = (k + 1) - len(pis)  # Target truncation
        if t_truncation > 0:
            pis += [pis[-1]] * t_truncation
            vs += [vs[-1]] * t_truncation
            rewards += [rewards[-1]] * t_truncation

        # One hot encode actions.
        enc_actions = np.zeros([len(actions), self.game.getActionSize()])
        enc_actions[np.arange(len(actions)), actions] = 1

        return enc_actions, (np.array(vs), np.array(rewards), np.array(pis))  # (Actions, Targets)

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
            self.game.buildTrajectory(histories[c[0]], None, None, self.neural_net.net_args.observation_length, t=c[1]),
            *self.buildHypotheticalSteps(histories[c[0]], c[1], k=self.args.K),
            loss_scale
        )
            for c, loss_scale in zip(sample_coordinates, sample_weight)
        ]

        return examples

    def executeEpisode(self) -> GameHistory:
        """
        This function executes one episode of self-play, starting with player 1.

        It uses a temp=1 if episode_step < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            history: A data structure containing all observed states and statistics
                     from the perspective of the past current players.
                     The structure is of the form (s_t, a_t, player_t, pi_t, r_t, v_t, z_t)
        """
        history = GameHistory()
        s = self.game.getInitialState()
        current_player = episode_step = temp = 1

        while not self.game.getGameEnded(s, current_player):  # Boardgames: If loop ends => current player lost
            # Turn action selection to greedy as an episode progresses.
            if episode_step % self.args.tempThreshold == 0:
                temp /= 2

            # Construct an observation array (o_1, ..., o_t).
            observation_array = self.game.buildTrajectory(history, s, current_player,
                                                          length=self.neural_net.net_args.observation_length)

            # Compute the move probability vector and state value using MCTS.
            pi, v = self.mcts.runMCTS(observation_array, temp=temp)

            # Take a step in the environment and observe the transition and store necessary statistics.
            action = np.random.choice(len(pi), p=pi)
            s_next, r, next_player = self.game.getNextState(s, action, 1)
            history.capture(s, action, current_player, pi, r, v)

            # Update state of control
            current_player = current_player if next_player == 1 else -current_player
            episode_step += 1

            s = self.game.getCanonicalForm(s_next, current_player)

        # TODO: Check whether the very last observation s needs to be stored (no play statistics?)
        # history.capture(s, -1, current_player, None, None, None)

        # Compute z_t for each observation. N-step returns for general MDPs or game outcomes for boardgames
        history.compute_returns(gamma=self.args.gamma, n=(self.args.n if not self.args.boardgame else None))

        return history

    def selfPlay(self) -> None:
        iteration_train_examples = list()

        eps_time = AverageMeter()
        bar = Bar('Self Play', max=self.args.numEps)
        end = time.time()

        for eps in range(self.args.numEps):
            self.mcts.clear_tree()  # Reset the search tree after every game.
            iteration_train_examples.append(self.executeEpisode())

            if sum(map(len, iteration_train_examples)) > self.args.maxlenOfQueue:
                iteration_train_examples.pop(0)

            # Bookkeeping + plot progress
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = f'({eps}/{self.args.numEps}) Eps Time: {eps_time.avg:.3f}s | ' \
                         f'Total: {bar.elapsed_td:} | ETA: {bar.eta_td:}'
            bar.next()
        bar.finish()

        # Store data from previous self-play iterations into the history
        self.trainExamplesHistory.append(iteration_train_examples)

    def backpropagation(self, history: typing.List[GameHistory]) -> None:
        eps_time = AverageMeter()
        bar = Bar('Backpropagation', max=self.args.numTrainingSteps)

        end = time.time()
        for epoch in range(self.args.numTrainingSteps):
            batch = self.sampleBatch(history)

            # Backpropagation
            loss = self.neural_net.train(batch)

            # Bookkeeping + plot progress
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = f'({epoch + 1}/{self.args.numTrainingSteps}) Eps Time: {eps_time.avg:.3f}s | ' \
                         f'Total: {bar.elapsed_td} | ETA: {bar.eta_td} | loss: {loss:.4f}'
            bar.next()
        bar.finish()

    def learn(self) -> None:
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it trains the neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        Afterwards the current neural network weights are stored and the loop continues.
        """
        for i in range(1, self.args.numIters + 1):
            print(f'------ITER {i}------')

            # Gather training data.
            self.selfPlay()

            n = len(self.trainExamplesHistory)
            print(f"Replay buffer filled with data from {n} self play iterations, at "
                  f"{100 * n / self.args.numItersForTrainExamplesHistory}% of maximum capacity.")

            # Backup history to a file
            self.saveTrainExamples(i - 1)

            # Flatten examples over self-play episodes and sample a training batch.
            complete_history = list()
            for episode_history in self.trainExamplesHistory:
                complete_history += episode_history

            self.backpropagation(complete_history)

            print('Storing a snapshot of the new model')
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.args.load_folder_file[-1])

    def saveTrainExamples(self, iteration: int) -> None:
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

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
