"""
TODO Revamp to work with single player games. Prioritized sampling.
"""
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import trange

from Arena import Arena
from AlphaZero.MCTS import MCTS


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, neural_net, args) -> None:
        self.game = game
        self.neural_net = neural_net
        self.opponent_net = self.neural_net.__class__(self.game, neural_net.net_args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.neural_net, self.args)
        self.trainExamplesHistory = []  # history of the most recent examples from args.numItersForTrainExamplesHistory
        self.skipFirstSelfPlay = False  # can be overridden in loadTrainExamples()
        self.observation_encoding = game.Representation.CANONICAL  # TODO

    @staticmethod
    def getCheckpointFile(iteration: int) -> str:
        return f'checkpoint_{iteration}.pth.tar'

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
        train_examples = []
        state = self.game.getInitialState()
        current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_state = self.game.getCanonicalForm(state, current_player)  # Flip
            temp = int(episode_step < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonical_state, temp=temp)
            sym = self.game.getSymmetries(canonical_state, pi)
            for b, p in sym:
                train_examples.append([b, current_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            state, reward, current_player = self.game.getNextState(state, action, current_player)  # FLip

            r = self.game.getGameEnded(state, current_player)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != current_player))) for x in train_examples]

    def learn(self) -> None:
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            print(f'------ITER {i}------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iteration_train_examples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in trange(self.args.numEps, desc="Self Play", file=sys.stdout):
                    self.mcts = MCTS(self.game, self.neural_net, self.args)  # reset search tree
                    iteration_train_examples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iteration_train_examples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print(f"length data buffer = {len(self.trainExamplesHistory)} => remove the oldest train_examples")
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
            opponent_mcts = MCTS(self.game, self.opponent_net, self.args)

            self.neural_net.train(train_examples)
            network_mcts = MCTS(self.game, self.neural_net, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(self.game,
                          lambda x: np.argmax(opponent_mcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(network_mcts.getActionProb(x, temp=0)))
            losses, wins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (wins, losses, draws))
            if losses + wins == 0 or float(wins) / (losses + wins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.neural_net.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile())
                self.neural_net.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def saveTrainExamples(self, iteration: int) -> None:
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

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
            self.skipFirstSelfPlay = True
