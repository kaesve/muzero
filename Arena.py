import typing
import sys

import numpy as np
from tqdm import trange

from Game import Game
from utils.selfplay_utils import GameHistory
from Experimenter.Players import Player


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, game: Game, player1: Player, player2: Player = None,
                 display: typing.Callable = None) -> None:
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.trajectories = [GameHistory(), GameHistory()]

    def playGame(self, verbose: bool = False) -> int:
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        all([x.refresh() for x in self.trajectories])
        players = [self.player2, None, self.player1]
        cur_player = 1
        state = self.game.getInitialState()
        it = 0

        while not self.game.getGameEnded(state, cur_player):
            it += 1

            if verbose:
                print(f"Turn {it} Player {cur_player}")
                self.display(state)

            action = players[cur_player + 1].act(state, cur_player)

            valid_moves = self.game.getLegalMoves(self.game.getCanonicalForm(state, cur_player), 1)
            if valid_moves[action] == 0:
                action = len(valid_moves)  # Resign.

            # Ensure that the opponent also observes the environment
            players[1 - cur_player].capture(state, action, cur_player)

            state, r, cur_player = self.game.getNextState(state, action, cur_player)

        if verbose:
            print(f"Game over: Turn {it}Result {self.game.getGameEnded(state, 1)}")
            self.display(state)

        return cur_player * self.game.getGameEnded(state, cur_player)

    def playTrial(self, num_trials: int, verbose: bool = False) -> np.ndarray:
        pass  # TODO: e.g. Atari

    def playGames(self, num_games: int, verbose: bool = False) -> typing.Tuple[int, int, int]:
        """
        Plays 2 * num_games games such that player 1 and 2 start an uniform number of times.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws:  games won by nobody
        """
        results = list()

        for _ in range(num_games):
            results.append(self.playGame(verbose=verbose))

        one_won = np.sum(np.array(results) == 1).item()
        two_won = np.sum(np.array(results) == -1).item()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num_games):
            results.append(self.playGame(verbose=verbose))

        one_won += np.sum(np.array(results) == -1).item()
        two_won += np.sum(np.array(results) == 1).item()

        return one_won, two_won, (one_won + two_won - num_games * 2)
