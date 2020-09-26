import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np

from Arena import Arena
from AlphaZero.MCTS import MCTS
from utils import Bar, AverageMeter


class MuZeroCoach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, neural_net, args):
        self.game = game
        self.neural_net = neural_net
        self.opponent_net = self.neural_net.__class__(self.game, neural_net.net_args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.neural_net, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory last iterations
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
            train_examples: a list of examples of the form (canonical_state, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_examples = list()
        s = self.game.getInitBoard()
        self.current_player = 1
        episode_step = 0
        z = 0

        while z == 0:
            episode_step += 1
            canonical_state = self.game.getCanonicalForm(s, self.current_player)

            # Turns to greedy selection after exceeding step threshold
            temp = int(episode_step < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonical_state, temp=temp)
            action = np.random.choice(len(pi), p=pi)

            s_next, r, self.current_player = self.game.getNextState(s, action, self.current_player)  # flip
            train_examples.append([canonical_state, self.current_player, pi, r, None])

            z = self.game.getGameEnded(s_next, self.current_player)
            s = s_next

        return [(x[0], x[2], +x[3], +z) if x[1] == self.current_player else  # +reward and +(n-step return)
                (x[0], x[2], -x[3], -z) for x in train_examples]             # -reward and -(n-step return)

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        # TODO:
        #   - Prioritized sampling for non zerosum games.
        #   - Constructing observation array (o_1, ..., o_t) from history
        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iteration_train_examples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.neural_net, self.args)  # reset search tree
                    iteration_train_examples += self.executeEpisode()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg,
                        total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iteration_train_examples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                      " => remove the oldest train_examples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the AlphaZeroModel from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            train_examples = []
            for e in self.trainExamplesHistory:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.opponent_net.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.opponent_net, self.args)

            self.neural_net.train(train_examples)
            nmcts = MCTS(self.game, self.neural_net, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.neural_net.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

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
