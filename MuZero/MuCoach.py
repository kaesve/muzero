import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler

import numpy as np

from AlphaZero.MCTS import MCTS
from utils import Bar, AverageMeter


class MuZeroCoach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    class GameHistory:

        def __init__(self):
            self.state_history = list()
            self.player_history = list()
            self.action_history = list()

        def capture(self, state, action, player):
            self.state_history.append(state)
            self.action_history.append(action)
            self.player_history.append(player)

        def refresh(self):
            self.state_history = list()
            self.player_history = list()
            self.action_history = list()

    def __init__(self, game, neural_net, args):
        self.game = game
        self.neural_net = neural_net
        self.opponent_net = self.neural_net.__class__(self.game, neural_net.net_args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.neural_net, self.args)
        self.trainExamplesHistory = deque(maxlen=self.args.numItersForTrainExamplesHistory)
        self.skipFirstSelfPlay = False  # can be overridden in loadTrainExamples()
        self.current_player = 1

    @staticmethod
    def getCheckpointFile(iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temp=1 if episode_step < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            train_examples: a list of examples of the form (canonical_state, currPlayer, pi, r, v)
                           pi is the MCTS informed policy vector, r and v are the reward and
                           n-step returns (game outcomes for boardgames). If the last player equals
                           currPlayer, then r and v are positive, otherwise they are negated.
        """

        history, train_examples = list(), list()
        s = self.game.getInitBoard()
        self.current_player = 1
        episode_step = 0

        while not self.game.getGameEnded(s, self.current_player):
            episode_step += 1
            observation = self.game.buildTrajectory(history)
            temp = int(episode_step < self.args.tempThreshold)

            pi = self.mcts.getActionProb(observation, temp=temp)
            action = np.random.choice(len(pi), p=pi)

            s_next, r, self.current_player = self.game.getNextState(s, action, self.current_player)
            
            train_examples.append([canonical_state, self.current_player, pi, r, None])
            history.append((s_next, action, self.current_player))
            
            s = s_next

        # TODO: n step return computation
        return [(x[0], x[2], +x[3], +1) if x[1] == self.current_player else  # +reward and +(n-step return)
                (x[0], x[2], -x[3], -1) for x in train_examples]             # -reward and -(n-step return)

    def selfPlay(self):
        iteration_train_examples = deque([], maxlen=self.args.maxlenOfQueue)

        eps_time = AverageMeter()
        bar = Bar('Self Play', max=self.args.numEps)
        end = time.time()

        for eps in range(self.args.numEps):
            self.mcts = MCTS(self.game, self.neural_net, self.args)  # Reset the search tree
            iteration_train_examples += self.executeEpisode()

            # Bookkeeping + plot progress
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
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

            if not self.skipFirstSelfPlay or i > 1:
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

            # Make copy network for the opponent (TODO: Direct keras getweights/ setweights memory leak?)
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
            self.skipFirstSelfPlay = True
