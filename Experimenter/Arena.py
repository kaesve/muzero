"""
Defines a class to handle pitting of agents on a provided environment.

Notes:
 - Code adapted from https://github.com/suragnair/alpha-zero-general
 - Base implementation done.
 - Documentation 15/11/2020
"""
import typing
import sys

import numpy as np
from tqdm import trange

from Games.Game import Game
from utils import DotDict
from utils.debugging import Monitor
import utils.debugging as debugging


class Arena:
    """
    An Arena class where any 2 agents can be pitted against each other.
    """

    def __init__(self, game: Game, player1, player2) -> None:
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

    def playTurnGame(self, first_player, second_player, verbose: bool = False) -> int:
        """
        Perform one turn-based game between players 1 and 2.

        :param first_player: Player interface for selecting actions. First to move.
        :param second_player: Player interface for selecting actions. Second to move.
        :param verbose: bool Whether to print out intermediary information about the state.
        :return: int Either 1 if player 1 won, -1 if player 2 won, or 0 if the game was a draw.
        """
        players = [second_player, None, first_player]
        cur_player = 1
        state = self.game.getInitialState()
        step = z = 0

        while not state.done:
            step += 1

            if verbose:
                print(f"Turn {step} Player {cur_player}")

            if debugging.RENDER:
                self.game.render(state)

            state.action = players[cur_player + 1].act(state)

            valid_moves = self.game.getLegalMoves(state)
            if not valid_moves[state.action]:
                state.action = len(valid_moves)  # Resign, will result in state.done = True

            # Capture an observation for both players
            players[cur_player + 1].observe(state)
            players[1 - cur_player].observe(state)

            cur_player = -cur_player
            state, _ = self.game.getNextState(state, state.action)
            z = self.game.getGameEnded(state, close=True)

        if verbose:
            print(f"Game over: Turn {step} Result {z}")

        return cur_player * z

    def playGame(self, player, verbose: bool = False) -> float:
        """
        Perform one single-player game with the provided player.

        :param player: Player interface for selecting actions.
        :param verbose: bool Whether to print out intermediary information about the state.
        :return: float The cumulative reward that the player managed to gather.
        """
        state = self.game.getInitialState()
        step = score = 0

        while not state.done:
            step += 1

            if verbose:
                print(f"Turn {step} Current score {score}")

            if debugging.RENDER:
                self.game.render(state)

            state.action = player.act(state)

            # Ensure that the opponent also observes the environment
            player.observe(state)
            state, r = self.game.getNextState(state, state.action)
            score += r

            # Closes the environment if terminated.
            _ = self.game.getGameEnded(state, close=True)

        if verbose:
            print(f"Game over: Step {step} Final score {score}")

        return score

    def playGames(self, num_trials: int, verbose: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Perform a series of games for both player 1 and 2 and return their accumulated scores.
        :param num_trials: int Number of trials to play for both player 1 and 2.
        :param verbose: bool Whether to print out intermediary state information during play.
        :return: tuple (np.ndarray, np.ndarray) Accumulated scores for player 1 and 2, respectively.
        """
        p1_scores, p2_scores = list(), list()

        for _ in trange(num_trials, desc="Pitting", file=sys.stdout):
            self.player1.refresh()
            self.player2.refresh()

            p1_scores.append(self.playGame(self.player1, verbose=verbose))
            p2_scores.append(self.playGame(self.player2, verbose=verbose))

        return np.array(p1_scores), np.array(p2_scores)

    def playTurnGames(self, num_games: int, verbose: bool = False) -> typing.Tuple[int, int, int]:
        """
        Perform a series of turn based games for player 1 and 2 and return their performance.
        Each player has equal opportunity to start first in the environment by performing 2 * num_games episodes.
        :param num_games: int Half the number of games to play between player 1 and 2.
        :param verbose: bool Whether to print out intermediary state information during play.
        :return: tuple (int, int, int) Number of: Wins player 1, Wins player 2, Draws.
        """
        results = list()
        for _ in trange(num_games, desc="Pitting Player 1 first", file=sys.stdout):
            self.player1.refresh()
            self.player2.refresh()

            results.append(self.playTurnGame(self.player1, self.player2, verbose=verbose))

        one_won = np.sum(np.array(results) == 1).item()
        two_won = np.sum(np.array(results) == -1).item()

        results = list()
        for _ in trange(num_games, desc="Pitting Player 2 first", file=sys.stdout):
            self.player1.refresh()
            self.player2.refresh()

            results.append(self.playTurnGame(self.player2, self.player1, verbose=verbose))

        one_won += np.sum(np.array(results) == -1).item()
        two_won += np.sum(np.array(results) == 1).item()

        return one_won, two_won, (one_won + two_won - num_games * 2)

    def pitting(self, args: DotDict, logger: Monitor) -> bool:
        """
        Pit player1 and player2 against each other in the provided environment to determine the better player.
        :param args: DotDict Data structure containing parameters for evaluating which player won the pitting-trials.
        :param logger: debugging.Monitor Class to log output information to.
        :return: bool Whether player 1 was found to be the better player conditional on the provided parameters.
        """
        print("Pitting players...")

        if self.game.n_players == 1:
            p1_score, p2_score = self.playGames(args.pitting_trials)

            wins, draws = np.sum(p1_score > p2_score), np.sum(p1_score == p2_score)
            losses = args.pitting_trials - (wins + draws)

            logger.log(p1_score.mean(), "Average Trial Reward")
            logger.log_distribution(p1_score, "Trial Reward")

            print(f'AVERAGE PLAYER 1 SCORE: {p1_score.mean()} ; AVERAGE PLAYER 2 SCORE: {p2_score.mean()}')
        else:
            losses, wins, draws = self.playTurnGames(args.pitting_trials)

        print(f'CHAMPION/CONTENDER WINS : {wins} / {losses} ; DRAWS : {draws} ; '
              f'NEW CHAMPION ACCEPTANCE RATIO : {args.pit_acceptance_ratio}')

        return losses + wins > 0 and wins / (losses + wins) >= args.pit_acceptance_ratio
