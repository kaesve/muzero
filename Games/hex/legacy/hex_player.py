"""
This Python file defines the Class for a hex-game Player. This is essentially
a wrapper for intuitive representation of a player. The Class is written in
such a way that the formal representation of Markov Decision Process Control
is made evident --- i.e., given a state s, evaluate the policy pi(s) to get
an action a = pi(s).

:version: FINAL
:date: 07-02-2020
:author: Joery de Vries and Ken Voskuil
:edited by: Joery de Vries and Ken Voskuil
"""


class Player(object):

    def __init__(self, policy, player):
        """
        Class to represent a hex-game Player. The player is defined by its
        assigned color and the policy that the player follows.
        For example: Player RED follows a Minimax policy.
        :param policy: Policy Derived Class that implements a move-choosing algorithm .
        :param player: int HexBoard player color.
        :see: .hex_policies
        """
        self._policy = policy
        self._player = player

    @property
    def color(self):
        """
        Get the color of the player.
        :return: int Color of the player
        :see: HexBoard in .hex_skeleton
        """
        return self._player

    def switch(self, color):
        """
        Italy when loosing: /switchteams
        Set internal parameters to play as the other team
        :param color: int HexBoard player color to switch to
        """
        self._player = color
        self._policy.set_perspective(color)

    def select_move(self, state):
        """
        With a given state s call the policy function pi(s) to
        generate an action a = pi(s), and return this action.

        (Calls the policy on a board and returns a chosen move.)
        :param state: State of the current environment.
        :return: The move to make on the board.
        :see: .hex_policies
        """
        return self._policy.generate_move(state)
