import typing
import sys

import numpy as np
from tqdm import trange

from Game import Game


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, game: Game, player1: typing.Callable, player2: typing.Callable = None,
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

    def playGame(self, verbose: bool = False) -> int:
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        cur_player = 1
        state = self.game.getInitialState()
        it = 0

        while not self.game.getGameEnded(state, cur_player):
            it += 1

            if verbose:
                print("Turn ", str(it), "Player ", str(cur_player))
                self.display(state)

            action = players[cur_player + 1](self.game.buildObservation(state, cur_player))

            valid_moves = self.game.getLegalMoves(self.game.getCanonicalForm(state, cur_player), 1)

            if valid_moves[action] == 0:
                action = len(valid_moves)  # Resign.

            state, r, cur_player = self.game.getNextState(state, action, cur_player)

        if verbose:
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(state, 1)))
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

        for _ in trange(num_games, desc="player 1 to move first", file=sys.stdout):
            results.append(self.playGame(verbose=verbose))

        one_won = np.sum(np.array(results) == 1).item()
        two_won = np.sum(np.array(results) == -1).item()

        self.player1, self.player2 = self.player2, self.player1

        for _ in trange(num_games, desc="player 2 to move first", file=sys.stdout):
            results.append(self.playGame(verbose=verbose))

        one_won += np.sum(np.array(results) == -1).item()
        two_won += np.sum(np.array(results) == 1).item()

        return one_won, two_won, (one_won + two_won - num_games * 2)
