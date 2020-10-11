import numpy as np

from utils.selfplay_utils import GameHistory
from utils import DotDict


class Player:

    def __init__(self, game, parametric: bool = False) -> None:
        self.game = game
        self.parametric = parametric
        self.history = GameHistory()
        self.name = ""

    def bind_history(self, history: GameHistory):
        self.history = history

    def capture(self, state: np.ndarray, action: int, player: int) -> None:
        pass

    def act(self, state: np.ndarray, player: int) -> int:
        pass


class AlphaZeroPlayer(Player):

    def __init__(self, game, search_engine, model, config: DotDict) -> None:
        super().__init__(game, parametric=True)
        self.search_engine = search_engine
        self.model = model
        self.config = config
        self.name = config.name

    def act(self, state: np.ndarray, player: int) -> int:
        pi = self.search_engine.getActionProb(state, temp=0)
        return np.argmax(pi).item()


class MuZeroPlayer(Player):

    def __init__(self, game, search_engine, model, config: DotDict) -> None:
        super().__init__(game, parametric=True)
        self.search_engine = search_engine
        self.model = model
        self.config = config
        self.name = config.name

    def capture(self, state: np.ndarray, action: int, player: int) -> None:
        o_t = self.game.buildObservation(state, player, form=self.game.Observation.HEURISTIC)
        self.history.capture(o_t, action, player, np.array([]), 0, 0)

    def act(self, state: np.ndarray, player: int) -> int:
        o_t = self.game.buildObservation(state, player, form=self.game.Observation.HEURISTIC)
        stacked_observations = self.history.stackObservations(self.model.net_args.observation_length, o_t)

        root_actions = self.game.getLegalMoves(state, player)
        pi, _ = self.search_engine.runMCTS(stacked_observations, legal_moves=root_actions, temp=0)
        action = np.argmax(pi).item()

        self.history.capture(o_t, action, player, np.array([]), 0, 0)

        return np.argmax(pi).item()


class RandomPlayer(Player):
    name: str = "Random"

    def act(self, state, player):
        mass_valid = self.game.getLegalMoves(state, player)
        return np.random.choice(len(mass_valid), p=mass_valid / np.sum(mass_valid))


class DeterministicPlayer(Player):
    name: str = "Deterministic"

    def act(self, state: np.ndarray, player: int) -> int:
        mass_valid = self.game.getLegalMoves(state, player)
        indices = np.where(mass_valid == 1)
        return indices[0]


class ManualPlayer(Player):

    def __init__(self, game, name):
        super().__init__(game)
        self.name = name

    def act(self, state: np.ndarray, player: int) -> int:
        mass_valid = self.game.getLegalMoves(state, player)
        indices = np.where(mass_valid == 1)

        move = None
        while move is None:
            move_str = input("Input an integer indicating a move:")
            if move_str.isdigit():
                if int(move_str) in indices:
                    move = int(move_str)

        return move

