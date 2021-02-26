"""
This file defines functions for estimating the 'skill' of different
game-playing policies.

The default_experiment performs an example tournament between three
fixed policies -- serving as an example implementation for other experiments.

:version: FINAL
:date: 07-02-2020
:author: Joery de Vries and Ken Voskuil
:edited by: Joery de Vries and Ken Voskuil
:references: https://trueskill.org/
"""

from itertools import permutations, combinations

import numpy as np
from trueskill import Rating, rate_1vs1

from .hex_player import Player
from .hex_game import playgame
from Games.hex.HexLogic import HexBoard
from .hex_policies import MinimaxPolicy
from .hex_heuristics import RandomHeuristic, DijkstraHeuristic


def doubles_ratings(boardsize, policy_list, resolution=10, verbose=False, name_list=None, monitor=False, ratings=None):
    """
    Compute individual ELO ratings for each policy in the policy list
    on the hex-game with the provided boardsize. The ELO ratings are calculated
    by performing pair-wise tournament matches with each possible policy.

    In contrast to permuted_ratings this function updates the rating after two policies
    have played as either the RED or BLUE team (doubles). If policy 1 wins as RED and loses as BLUE,
    then it has drawn with policy 2. If policy 1 wins as RED and BLUE, it has won.


    This procedure is repeated 'resolution' times.
    :param boardsize: int Size of the board to play on.
    :param policy_list: list List of different policies to test.
    :param resolution: int Amount of times to repeat the pair-wise tournament.
    :param name_list: list List of strings that contain policy names to print out.
    :param monitor: bool Whether to track the progress of ratings after each game.
    :param verbose: boolean Whether to print out current progress.
    :param ratings: list List of TrueSkill Rating objects, if None: fresh ones are generated.
    :return: list A list of Rating objects for each policy. (optionally the rating history dict)
    :see: Rating, rate_1vs1
    :see: permuted_ratings
    :references: https://trueskill.org/
    """
    history = dict()
    pairs = list(combinations(range(len(policy_list)), r=2))

    if not ratings:  # If ratings are available --> re-use them.
        ratings = [Rating() for _ in range(len(policy_list))]

    if monitor:
        history = {i: [(ratings[i].mu, ratings[i].sigma)] for i in range(len(policy_list))}

    for _ in range(resolution):
        if verbose:
            print("Tourney repetition: ", _)

        np.random.shuffle(pairs)  # Not strictly neccesary

        for i, (pol_1, pol_2) in enumerate(pairs):
            if verbose:
                print("Pair {} of {}  ".format(i, len(pairs), end='\r'))
                if name_list:
                    print("Playing {} against {}".format(name_list[pol_1], name_list[pol_2]))

            policy1, policy2 = policy_list[pol_1], policy_list[pol_2]

            # For ease of reference, take p1 as red and p2 as blue.
            # Create new Players using the corresponding Policies.
            policy1.set_perspective(HexBoard.RED)
            red = Player(policy1, HexBoard.RED)

            policy2.set_perspective(HexBoard.BLUE)
            blue = Player(policy2, HexBoard.BLUE)

            result = playgame(boardsize, red=red, blue=blue, doubles=True, verbose=verbose)

            if result == HexBoard.RED:
                ratings[pol_1], ratings[pol_2] = rate_1vs1(ratings[pol_1], ratings[pol_2])
            elif result == HexBoard.BLUE:
                ratings[pol_2], ratings[pol_1] = rate_1vs1(ratings[pol_2], ratings[pol_1])
            else:  # draw
                ratings[pol_2], ratings[pol_1] = rate_1vs1(ratings[pol_2], ratings[pol_1], drawn=True)

            if monitor:
                history[pol_1].append((ratings[pol_1].mu, ratings[pol_1].sigma))
                history[pol_2].append((ratings[pol_2].mu, ratings[pol_2].sigma))

    if monitor:
        return ratings, history
    return ratings


def singles_ratings(boardsize, policy_list, resolution=10, verbose=False, name_list=None, monitor=False, ratings=None):
    """
    Compute individual ELO ratings for each policy in the policy list
    on the hex-game with the provided boardsize. The ELO ratings are calculated
    by performing pair-wise tournament matches with each possible policy. In
    other words all permutations of the policies are tried individually.

    In contrast to combined_ratings this function updates the rating after
    each individual game. A draw is, thus, not possible as games are played as
    single matches (only one perspective).

    This procedure repeated 'resolution' times.
    :param boardsize: int Size of the board to play on.
    :param policy_list: list List of different policies to test.
    :param resolution: int Amount of times to repeat the pair-wise tournament.
    :param verbose: boolean Whether to print out current progress.
    :param name_list: list List of strings that contain policy names to print out.
    :param monitor: bool Whether to track the progress of ratings after each game.
    :param ratings: list List of TrueSkill Rating objects, if None: fresh ones are generated.
    :return: list A list of Rating objects for each policy. (optionally the rating history dict)
    :see: Rating, rate_1vs1
    :see: combined_ratings
    :references: https://trueskill.org/
    """
    history = dict()
    tourney_pairs = list(permutations(range(len(policy_list)), r=2))

    if not ratings:  # If ratings are available --> re-use them.
        ratings = [Rating() for _ in range(len(policy_list))]

    if monitor:
        history = {i: [(ratings[i].mu, ratings[i].sigma)] for i in range(len(policy_list))}

    for _ in range(resolution):
        if verbose:
            print("Tourney repetition: ", _)

        np.random.shuffle(tourney_pairs)

        for i, (red_i, blue_i) in enumerate(tourney_pairs):
            if verbose:
                print("Pair {} of {}  ".format(i, len(tourney_pairs), end='\r'))
                if name_list:
                    print("Playing {} against {}".format(name_list[red_i], name_list[blue_i]))

            red_policy, blue_policy = policy_list[red_i], policy_list[blue_i]

            # Switch the internal perspective of the policy.
            red_policy.set_perspective(HexBoard.RED)
            blue_policy.set_perspective(HexBoard.BLUE)

            # Create new Players using the corresponding Policies.
            red = Player(red_policy, HexBoard.RED)
            blue = Player(blue_policy, HexBoard.BLUE)

            # 1v1 the two Players.
            result = playgame(boardsize, red=red, blue=blue, doubles=False, verbose=verbose)

            # Re-compute the 'ELO' ratings of the two player policies.
            if result == HexBoard.RED:
                ratings[red_i], ratings[blue_i] = rate_1vs1(ratings[red_i], ratings[blue_i])
            else:
                ratings[blue_i], ratings[red_i] = rate_1vs1(ratings[blue_i], ratings[red_i])

            if monitor:
                history[red_i].append((ratings[red_i].mu, ratings[red_i].sigma))
                history[blue_i].append((ratings[blue_i].mu, ratings[blue_i].sigma))

    if monitor:
        return ratings, history
    return ratings


def default_experiment(board_size, doubles=True, **kwargs):
    """
    Performs and records a default experiment on three different 'AI-agents':
    - Minimax policy with search depth 3 and a random evaluation heuristic.
    - Minimax policy with search depth 3 and a dijkstra-based evaluation heuristic.
    - Minimax policy with search depth 4 and a dijkstra-based evaluation heuristic.

    :param board_size: int How large the game-playing board should be.
    :param doubles: bool Whether to play singles or doubles matches (default doubles).
    :returns: list A list of ratings in order of the above three policies.
    :see: playgame in hex_game
    """
    names = [r'd=3 h=random', r'd=3 h=dijkstra', r'd=4 h=dijkstra']
    minmax_d3_rand = MinimaxPolicy(
        RandomHeuristic(-board_size ** 2, board_size ** 2), depth=3, itd=False, transpositions=False)
    minmax_d3_dijkstra = MinimaxPolicy(
        DijkstraHeuristic(), depth=3, itd=False, transpositions=False)
    minmax_d4_dijkstra = MinimaxPolicy(
        DijkstraHeuristic(), depth=4, itd=False, transpositions=False)

    policy_list = [minmax_d3_rand, minmax_d3_dijkstra, minmax_d4_dijkstra]

    if doubles:
        return doubles_ratings(board_size, policy_list, name_list=names, **kwargs)
    else:
        return singles_ratings(board_size, policy_list, name_list=names, **kwargs)
