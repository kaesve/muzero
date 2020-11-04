import typing
import sys

import numpy as np
from tqdm import trange

from Game import Game
from utils import DotDict
from utils.debugging import Monitor


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, game: Game, player1, player2,
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
        it = z = 0

        while not state.done:
            it += 1

            if verbose:
                print(f"Turn {it} Player {cur_player}")
                self.display(state)

            state.action = players[cur_player + 1].act(state)

            valid_moves = self.game.getLegalMoves(state)
            if not valid_moves[state.action]:
                state.action = len(valid_moves)  # Resign, will result in state.done = True

            # Capture an observation for both players
            players[cur_player + 1].observe(state)
            players[1 - cur_player].observe(state)

            cur_player = -cur_player
            state, _ = self.game.getNextState(state, state.action)
            z = self.game.getGameEnded(state)

        if verbose:
            print(f"Game over: Turn {it} Result {z}")
            self.display(state)

        return cur_player * z

    def playTrial(self, player, verbose: bool = False) -> float:
        state = self.game.getInitialState()
        it = score = 0

        while not state.done:
            it += 1

            if verbose:
                self.display(state)

            state.action = player.act(state)

            # Ensure that the opponent also observes the environment
            player.observe(state)
            state, r = self.game.getNextState(state, state.action)
            score += r

        if verbose:
            print(f"Game over: Step {it} Result {score}")
            self.display(state)

        return score

    def playTrials(self, num_trials: int, verbose: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:
        p1_scores, p2_scores = list(), list()

        for _ in trange(num_trials, desc="Pitting", file=sys.stdout):
            self.player1.refresh()
            self.player2.refresh()

            p1_scores.append(self.playTrial(self.player1, verbose=verbose))
            p2_scores.append(self.playTrial(self.player2, verbose=verbose))

        return np.array(p1_scores), np.array(p2_scores)

    def playGames(self, num_games: int, verbose: bool = False) -> typing.Tuple[int, int, int]:
        """
        Plays 2 * num_games games such that player 1 and 2 start an uniform number of times.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws:  games won by nobody
        """
        results = list()

        for _ in trange(num_games, desc="Pitting Player 1 first", file=sys.stdout):
            self.player1.refresh()
            self.player2.refresh()

            results.append(self.playGame(verbose=verbose))

        one_won = np.sum(np.array(results) == 1).item()
        two_won = np.sum(np.array(results) == -1).item()

        self.player1, self.player2 = self.player2, self.player1  # Swap

        results = list()
        for _ in trange(num_games, desc="Pitting Player 2 first", file=sys.stdout):
            self.player1.refresh()
            self.player2.refresh()

            results.append(self.playGame(verbose=verbose))

        one_won += np.sum(np.array(results) == -1).item()
        two_won += np.sum(np.array(results) == 1).item()

        self.player1, self.player2 = self.player2, self.player1  # Swap back.

        return one_won, two_won, (one_won + two_won - num_games * 2)

    def pitting(self, args: DotDict, logger: Monitor) -> bool:
        print("Pitting against previous version...")

        if self.game.n_players == 1:
            p1_score, p2_score = self.playTrials(args.pitting_trials)

            wins, draws = np.sum(p1_score > p2_score), np.sum(p1_score == p2_score)
            losses = args.pitting_trials - (wins + draws)

            logger.log(p1_score.mean(), "Average Trial Reward")
            logger.log_distribution(p1_score, "Trial Reward")

            print(f'AVERAGE NEW SCORE: {p1_score.mean()} ; AVERAGE OLD SCORE: {p2_score.mean()}')
        else:
            losses, wins, draws = self.playGames(args.pitting_trials)

        print(f'NEW/OLD WINS : {wins} / {losses} ; DRAWS : {draws} ; ACCEPTANCE RATIO : {args.pit_acceptance_ratio}')

        return losses + wins > 0 and wins / (losses + wins) >= args.pit_acceptance_ratio
