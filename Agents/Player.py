"""
Defines interface and implementations of Player objects. This interface is used during trials/ tournaments and can
be used to pit algorithms against each other (MuZero Vs. AlphaZero for example).
"""
import typing
from abc import ABC, abstractmethod

import numpy as np

from utils.game_utils import GameState
from utils.selfplay_utils import GameHistory
from utils import DotDict

from AlphaZero.implementations.DefaultAlphaZero import DefaultAlphaZero
from AlphaZero.AlphaMCTS import MCTS as AlphaZeroMCTS
from MuZero.implementations.DefaultMuZero import DefaultMuZero
from MuZero.implementations.BlindMuZero import BlindMuZero
from MuZero.MuMCTS import MuZeroMCTS


class Player(ABC):
    """ Interface for players for general environment control/ game playing. """

    def __init__(self, game, arg_file: typing.Optional[str] = None, name: str = "", parametric: bool = False) -> None:
        """
        Initialization of the Base Player object.
        :param game: Game Instance of Games.Game that implements environment logic.
        :param arg_file: str Path to JSON configuration file for the agent/ player.
        :param name: str Name to annotate this player with (useful during tournaments)
        :param parametric: bool Whether the agent depends on parameters or is parameter-free.
        """
        self.game = game
        self.player_args = arg_file
        self.parametric = parametric
        self.histories = list()
        self.history = GameHistory()
        self.name = name

    def bind_history(self, history: GameHistory) -> None:
        """ Bind an external memory object for keeping track of environment observations. """
        self.history = history

    def refresh(self, hard_reset: bool = False) -> None:
        """ Refresh or reinitialize memory/ observation trajectory of the agent. """
        if hard_reset:
            self.histories = list()
            self.history.refresh()
        else:
            self.histories.append(self.history)
            self.history = GameHistory()

    def observe(self, state: GameState) -> None:
        """ Capture an environment state observation within the agent's memory. """
        self.history.capture(state, np.array([]), 0, 0)

    def clone(self):
        """ Create a new instance of this Player object using equivalent parameterization """
        return self.__class__(self.game, self.player_args)

    @abstractmethod
    def act(self, state: GameState) -> int:
        """
        Method that should be implemented as an agent-specific action-selection method.
        :param state: GameState Data structure containing the specifics of the current environment state.
        :return: int Integer action to be performed in the environment.
        """


class DefaultAlphaZeroPlayer(Player):
    """
    Standard AlphaZero agent that samples actions from MCTS within a given environment model.
    """

    def __init__(self, game, arg_file: typing.Optional[str] = None, name: str = "") -> None:
        super().__init__(game, arg_file, name, parametric=True)
        if self.player_args is not None:
            # Initialize AlphaZero by loading its parameter config and constructing the network and search classes.
            self.args = DotDict.from_json(self.player_args)

            self.model = DefaultAlphaZero(self.game, self.args.net_args, self.args.architecture)
            self.search_engine = AlphaZeroMCTS(self.game, self.model, self.args.args)
            self.name = self.args.name

    def set_variables(self, model, search_engine, name: str) -> None:
        """ Assign Neural Network and Search class to an external reference """
        self.model = model
        self.search_engine = search_engine
        self.name = name

    def refresh(self, hard_reset: bool = False):
        """ Refresh internal state of the Agent along with stored statistics within the MCTS tree """
        super().refresh()
        self.search_engine.clear_tree()

    def act(self, state: GameState) -> int:
        """ Sample actions using MCTS using the given environment model. """
        pi, _ = self.search_engine.runMCTS(state, self.history, temp=0)
        return np.argmax(pi).item()


class DefaultMuZeroPlayer(Player):
    """
    Standard MuZero agent that samples actions from MCTS within its learned model.
    """

    def __init__(self, game, arg_file: typing.Optional[str] = None, name: str = "") -> None:
        super().__init__(game, arg_file, name, parametric=True)
        if self.player_args is not None:
            # Initialize MuZero by loading its parameter config and constructing the network and search classes.
            self.args = DotDict.from_json(self.player_args)

            self.model = DefaultMuZero(self.game, self.args.net_args, self.args.architecture)
            self.search_engine = MuZeroMCTS(self.game, self.model, self.args.args)
            self.name = self.args.name

    def set_variables(self, model, search_engine, name: str) -> None:
        """ Assign Neural Network and Search class to an external reference """
        self.model = model
        self.search_engine = search_engine
        self.name = name

    def refresh(self, hard_reset: bool = False) -> None:
        """ Refresh internal state of the Agent along with stored statistics within the MCTS tree """
        super().refresh()
        self.search_engine.clear_tree()

    def act(self, state: GameState) -> int:
        """ Samples actions using MCTS within the learned RNN/ MDP model. """
        pi, _ = self.search_engine.runMCTS(state, self.history, temp=0)
        return np.argmax(pi).item()


class BlindMuZeroPlayer(Player):
    """
    MuZero agent that receives observations according to a (sparse) schedule. If no observations are provided, the
    agent must extrapolate future time steps within its learned model.
    """

    def __init__(self, game, nested_config: typing.Optional[DotDict] = None, name: str = "") -> None:
        super().__init__(game, nested_config.file, name, parametric=True)
        if self.player_args is not None:
            # Initialize MuZero by loading its parameter config and constructing the network and search classes.
            # Additionally assign/ bind internal MDP memory to enable planning strictly within the learned model.
            self.args = DotDict.from_json(self.player_args)

            self.model = BlindMuZero(self.game, self.args.net_args, self.args.architecture, nested_config.refresh_freq)
            self.model.bind(self.history.actions)

            self.search_engine = MuZeroMCTS(self.game, self.model, self.args.args)
            self.name = self.args.name

    def set_variables(self, model, search_engine, name: str) -> None:
        """ Assign Neural Network and Search class to an external reference """
        self.model = model
        self.search_engine = search_engine
        self.name = name

    def refresh(self, hard_reset: bool = False) -> None:
        """
        Refresh internal state of the Agent along with stored statistics within the MCTS tree.
        Additionally refreshes the current state within the MuZero learned MDP to again start from an observation's
        embedding.
        """
        super().refresh()
        self.search_engine.clear_tree()
        # Reset trajectory within the learned model back to the embedding function.
        self.model.reset()
        self.model.bind(self.history.actions)

    def act(self, state: GameState) -> int:
        """
        Sample actions from MCTS. Although the 'state' is provided, MCTS does not start from the embedding of the
        current observation (only when refreshed by the schedule) but starts from the last step within the
        learned model trajectory.
        """
        pi, _ = self.search_engine.runMCTS(state, self.history, temp=0)
        return np.argmax(pi).item()


class RandomPlayer(Player):
    """ Agent to act uniformly random within some environment """
    name: str = "Random"

    def act(self, state: GameState) -> int:
        """ Randomly select some legal action. """
        mass_valid = self.game.getLegalMoves(state)
        return np.random.choice(len(mass_valid), p=mass_valid / np.sum(mass_valid))


class DeterministicPlayer(Player):
    """ Agent to always select the first legal action within an environment """
    name: str = "Deterministic"

    def act(self, state: GameState) -> int:
        """ Select actions based on the first encountered legal action within a flattened action array. """
        mass_valid = self.game.getLegalMoves(state)
        indices = np.ravel(np.where(mass_valid == 1))
        return indices[0]


class ManualPlayer(Player):
    """ Agent to be played by an external user based on console input """
    name: str = "Manual"

    def __init__(self, game, config: typing.Optional[str] = None) -> None:
        """ Initialize player and poll for the agent's name. """
        super().__init__(game, config)
        self.name = input("Input a player name: ")

    def act(self, state: GameState) -> int:
        """ Select action based on user input. """
        mass_valid = self.game.getLegalMoves(state)
        indices = np.ravel(np.where(mass_valid == 1))

        move = None
        while move is None:
            print("Available actions:", indices)
            move_str = input("Input an integer indicating a move:")
            if move_str.isdigit() and int(move_str) in indices:
                move = int(move_str)

        return move

