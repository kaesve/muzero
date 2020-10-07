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
from utils import Bar, AverageMeter
from utils.storage import DotDict


class MuZeroCoach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    class GameHistory:
        """
        Data container for keeping track of games
        """

        def __init__(self) -> None:
            """

            """
            self.states = self.players = self.actions = self.probabilities = \
                self.rewards = self.predicted_returns = self.actual_returns = None
            self.refresh()

        def __len__(self) -> int:
            """Get length of current stored trajectory"""
            return len(self.states)

        def capture(self, state: np.ndarray, action: int, player: int, pi: typing.List, r: float, v: float) -> None:
            """"""
            self.states.append(state)
            self.actions.append(action)
            self.players.append(player)
            self.probabilities.append(pi)
            self.rewards.append(r)
            self.predicted_returns.append(v)
            self.actual_returns.append(None)

        def refresh(self) -> None:
            """Clear all statistics within the class"""
            self.states, self.players, self.actions, self.probabilities, \
            self.rewards, self.predicted_returns, self.actual_returns = [[] for _ in range(7)]

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
        self.current_player = 1

    @staticmethod
    def getCheckpointFile(iteration: int) -> str:
        """"""
        return 'checkpoint_' + str(iteration) + '.pth.tar'

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
        vs = history.actual_returns[start:end+1]
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
        lengths = list(map(len, histories))
        n = self.neural_net.net_args.batch_size

        # Array of sampling probabilities over the range (0, total_data_length)
        sampling_probability = None
        sample_weight = np.ones(np.sum(lengths)) / n  # == Uniform weight update strength over batch.

        if self.args.prioritize:
            errors = np.array([np.abs(h.predicted_returns[i] - h.actual_returns[i])
                               for h in histories for i in range(len(h))])

            mass = np.pow(errors, self.args.prioritize_alpha)  # un-normalized mass
            sampling_probability = mass / np.sum(mass)

            # Adjust weight update strength proportionally to IS-ratio to prevent sampling bias.
            sample_weight = np.power(n * sampling_probability, -self.args.prioritize_beta)

        indices = np.random.choice(a=np.sum(lengths), size=self.neural_net.net_args.batch_size,
                                   replace=False, p=sampling_probability)

        # Map the flat indices to the correct histories and history indices.
        history_index_borders = np.cumsum([0] + lengths)
        history_indices = [(np.sum(i >= history_index_borders), i) for i in indices]

        # Of the form [(history_i, t), ...] \equiv history_it
        sample_coordinates = [(h_i - 1, i - history_index_borders[h_i-1]) for h_i, i in history_indices]

        # Construct training examples for MuZero of the form (input, action, (targets), loss_scalar)
        examples = [(
            self.game.buildTrajectory(histories[c[0]], None, None, self.neural_net.net_args.observation_length, t=c[1]),
            *self.buildHypotheticalSteps(histories[c[0]], c[1], k=self.args.K),
            sample_weight[lengths[c[0]] + c[1]]
        )
            for c in sample_coordinates
        ]

        return examples

    def computeReturns(self, history: GameHistory) -> None:  # TODO: Testing of this function.
        """

        :param history:
        :return:
        """
        # Update the MCTS estimate v_t with the more accurate estimates z_t
        if self.args.boardgame:
            # Boardgames
            for i in range(len(history)):
                history.actual_returns[i] = -1 if history.players[i] == self.current_player else 1
        else:
            # General MDPs. Letters follow notation from the paper.
            n = self.args.n_steps
            for t in range(len(history)):
                horizon = np.min([t + n, len(history)])
                discounted_rewards = [np.pow(self.args.gamma, k) * history.rewards[k] for k in range(t, horizon)]
                bootstrap = np.pow(self.args.gamma, horizon - t) * history.predicted_returns[horizon]
                history.actual_returns[t] = np.sum(discounted_rewards) + bootstrap

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
        history = self.GameHistory()
        s = self.game.getInitialState()
        self.current_player = 1
        episode_step = 1
        temp = 1

        while not self.game.getGameEnded(s, self.current_player):  # Boardgames: If loop ends => current player lost
            # Turn action selection to greedy as an episode progresses.
            if episode_step % self.args.tempThreshold == 0:
                temp /= 2

            # Construct an observation array (o_1, ..., o_t).
            observation_array = self.game.buildTrajectory(history, s, self.current_player,
                                                          length=self.neural_net.net_args.observation_length)

            # Compute the move probability vector and state value using MCTS.
            pi, v = self.mcts.runMCTS(observation_array, temp=temp)

            # Take a step in the environment and observe the transition and store necessary statistics.
            action = np.random.choice(len(pi), p=pi)  # TODO Check if action is in perspective of the canonicalForm
            s_next, r, next_player = self.game.getNextState(s, action, 1)
            history.capture(s, action, self.current_player, pi, r, v)

            # Update state of control
            self.current_player = self.current_player if next_player == 1 else -self.current_player
            episode_step += 1

            s = self.game.getCanonicalForm(s_next, self.current_player)

        # TODO: Check whether the very last observation s needs to be stored (no play statistics?)

        # Compute z_t for each observation. N-step returns for general MDPs or game outcomes for boardgames
        self.computeReturns(history)
        return history

    def selfPlay(self) -> None:
        """

        :return:
        """
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
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
        bar.finish()

        # Store data from previous self-play iterations into the history
        self.trainExamplesHistory.append(iteration_train_examples)

    def backprop(self, history: typing.List[GameHistory]) -> None:  # TODO: Tidy duplicate code.
        """

        :param history:
        :return:
        """
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
            bar.suffix = '({}/{}) Eps Time: {:.3f}s | Total: {} | ETA: {} | loss: {:.4f}'.format(
                epoch + 1, self.args.numTrainingSteps, eps_time.avg, bar.elapsed_td, bar.eta_td, loss)
            bar.next()
        bar.finish()

    def learn(self) -> None:
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(1, self.args.numIters + 1):
            print('------ITER {}------'.format(i))

            # Gather training data.
            self.selfPlay()

            n = len(self.trainExamplesHistory)
            print("Replay buffer filled with data from {} self play iterations, at {}% of maximum capacity.".format(
                n, 100 * n / self.args.numItersForTrainExamplesHistory))

            # Backup history to a file
            self.saveTrainExamples(i - 1)

            # Flatten examples over self-play episodes and sample a training batch.
            complete_history = list()
            for episode_history in self.trainExamplesHistory:
                complete_history += episode_history

            self.backprop(complete_history)

            print('Storing a snapshot of the new model')
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.args.load_folder_file[-1])

    def saveTrainExamples(self, iteration: int) -> None:
        """

        :param iteration:
        :return:
        """
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
            # examples based on the AlphaZeroModel were already collected (loaded)
