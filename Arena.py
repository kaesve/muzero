import time
import typing

from utils import Bar, AverageMeter
from Game import Game


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1: typing.Callable, player2: typing.Callable,
                 game: Game, display: typing.Callable = None) -> None:
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
        board = self.game.getInitialState()
        it = 0
        while self.game.getGameEnded(board, cur_player) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(cur_player))
                self.display(board)
            action = players[cur_player + 1](self.game.getCanonicalForm(board, cur_player))

            valids = self.game.getLegalMoves(self.game.getCanonicalForm(board, cur_player), 1)

            if valids[action] == 0:
                assert valids[action] > 0

            board, r, cur_player = self.game.getNextState(board, action, cur_player)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return cur_player * self.game.getGameEnded(board, cur_player)

    def playGames(self, num: int, verbose: bool = False) -> typing.Tuple[int, int, int]:
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        max_eps = int(num)

        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        for _ in range(num):
            game_result = self.playGame(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = f'({eps}/{max_eps}) Eps Time: {eps_time.avg:.3f}s | ' \
                         f'Total: {bar.elapsed_td:} | ETA: {bar.eta_td:}'
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            game_result = self.playGame(verbose=verbose)
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{max_eps}) Eps Time: {eps_time.avg:.3f}s | ' \
                         'Total: {bar.elapsed_td:} | ETA: {bar.eta_td:}'
            bar.next()

        bar.finish()

        return one_won, two_won, draws
