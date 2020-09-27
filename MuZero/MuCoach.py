import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler

import numpy as np

from MuZero.MuMCTS import MuZeroMCTS
from utils import Bar, AverageMeter


class MuZeroCoach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    class GameHistory:

        def __init__(self):
            self.states = list()
            self.players = list()
            self.actions = list()
            self.probabilities = list()
            self.rewards = list()
            self.predicted_returns = list()
            self.actual_returns = list()

        def __len__(self):
            return len(self.states)

        def capture(self, state, action, player, pi, r, v):
            self.states.append(state)
            self.actions.append(action)
            self.players.append(player)
            self.probabilities.append(pi)
            self.rewards.append(r)
            self.predicted_returns.append(v)
            self.actual_returns.append(None)

        def refresh(self):
            self.states, self.players, self.actions, self.probabilities, \
                self.rewards, self.predicted_returns, self.actual_returns = [[] for _ in range(7)]

    def __init__(self, game, neural_net, args):
        self.game = game
        self.neural_net = neural_net
        self.opponent_net = self.neural_net.__class__(self.game, neural_net.net_args)  # the competitor network
        self.args = args
        self.mcts = MuZeroMCTS(self.game, self.neural_net, self.args)
        self.trainExamplesHistory = deque(maxlen=self.args.numItersForTrainExamplesHistory)
        self.current_player = 1

    @staticmethod
    def getCheckpointFile(iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def computeReturns(self, history):  # TODO: Testing of this function.
        # Update the MCTS estimate v_t with the more accurate estimates z_t
        if self.args.zerosum:
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

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        training_statistics. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in training_statistics.

        It uses a temp=1 if episode_step < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            training_statistics: a list of examples of the form (canonical_state, currPlayer, pi, r, v)
                           pi is the MCTS informed policy vector, r and v are the reward and
                           n-step returns (game outcomes for boardgames). If the last player equals
                           currPlayer, then r and v are positive, otherwise they are negated.
        """
        history = self.GameHistory()
        s = self.game.getInitialState()
        self.current_player = 1
        episode_step = 1

        while not self.game.getGameEnded(s, self.current_player):  # Boardgames: If loop ends => current player lost
            # Turn action selection to greedy after exceeding the given threshold.
            temp = int(episode_step < self.args.tempThreshold)

            # Construct an observation array (o_1, ..., o_t).
            observation_array = self.game.buildTrajectory(history, s)

            # Compute the move probability vector and state value using MCTS.
            pi, v = self.mcts.runMCTS(observation_array, temp=temp)

            # Take a step in the environment and observe the transition and store necessary statistics.
            action = np.random.choice(len(pi), p=pi)
            s_next, r, next_player = self.game.getNextState(s, action, self.current_player)
            history.capture(s, action, self.current_player, pi, r, v)

            # Update state of control
            self.current_player = next_player
            episode_step += 1
            s = s_next

        # Compute z_t for each observation. N-step returns for general MDPs or game outcomes for boardgames
        self.computeReturns(history)
        return history

    def selfPlay(self):
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
                eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
        bar.finish()

        # Store data from previous self-play iterations into the history
        self.trainExamplesHistory.append(iteration_train_examples)

    def learn(self):
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
            print("Replay buffer filled with data from {} self play iterations, at {} of maximum capacity.".format(
                n, n / self.args.numItersForTrainExamplesHistory))

            # Backup history to a file
            self.saveTrainExamples(i - 1)

            # Extract all and shuffle examples before training
            complete_history = list()
            for episode_history in self.trainExamplesHistory:
                complete_history += episode_history
            np.random.shuffle(complete_history)

            # Backpropagation
            self.neural_net.train(complete_history)  # TODO: Complete history or a batch? Also: Prioritized sampling.

            print('Storing a snapshot of the new model')
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.args.load_folder_file[-1])

            # Make copy network for the opponent.
            self.opponent_net.load_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

        # Don't hog up storage space and clean up old (never to be used again) data.
        old_checkpoint = os.path.join(folder, self.getCheckpointFile(iteration-1) + '.examples')
        if os.path.isfile(old_checkpoint):
            os.remove(old_checkpoint)

    def loadTrainExamples(self):
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
