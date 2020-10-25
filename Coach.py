"""

"""
import os
import sys
from pickle import Pickler, Unpickler
from collections import deque

from utils import DotDict
from utils.selfplay_utils import GameHistory


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, args: DotDict) -> None:
        """

        :param game:
        :param args:
        """
        self.game = game
        self.args = args

        self.trainExamplesHistory = deque(maxlen=self.args.numItersForTrainExamplesHistory)

    @staticmethod
    def getCheckpointFile(iteration: int) -> str:
        return f'checkpoint_{iteration}.pth.tar'

    def executeEpisode(self) -> GameHistory:
        """
        This function executes one episode of self-play, starting with player 1.

        It uses a temp=1 if episode_step < tempThreshold, and thereafter uses temp=0.

        Returns:
            history: A data structure containing all observed states and statistics
                     from the perspective of the past current players.
                     The structure is of the form (s_t, a_t, player_t, pi_t, r_t, v_t, z_t)
        """
        pass

    def learn(self) -> None:
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it trains the neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        Afterwards the current neural network weights are stored and the loop continues.
        """
        pass

    def saveTrainExamples(self, iteration: int) -> None:
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)  # TODO: Change to h5py

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
