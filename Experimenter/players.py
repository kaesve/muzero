import numpy as np

from utils.game_utils import GameState
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

    def refresh(self) -> None:
        self.history.refresh()

    def observe(self, state: GameState) -> None:
        self.history.capture(state, np.array([]), 0, 0)

    def act(self, state: GameState) -> int:
        pass


class AlphaZeroPlayer(Player):

    def __init__(self, game, search_engine, model, config: DotDict) -> None:
        super().__init__(game, parametric=True)
        self.search_engine = search_engine
        self.model = model
        self.config = config
        self.name = config.name

    def refresh(self):
        super().refresh()
        self.search_engine.clear_tree()

    def act(self, state: GameState) -> int:
        pi, _ = self.search_engine.runMCTS(state, self.history, temp=0)
        return np.argmax(pi).item()


class MuZeroPlayer(Player):

    def __init__(self, game, search_engine, model, config: DotDict) -> None:
        super().__init__(game, parametric=True)
        self.search_engine = search_engine
        self.model = model
        self.config = config
        self.name = config.name

    def refresh(self):
        super().refresh()
        self.search_engine.clear_tree()

    def observe(self, state: GameState) -> None:
        self.history.capture(state, np.array([]), 0, 0)

    def act(self, state: GameState) -> int:
        pi, _ = self.search_engine.runMCTS(state, self.history, temp=0)
        return np.argmax(pi).item()


class RandomPlayer(Player):
    name: str = "Random"

    def act(self, state: GameState) -> int:
        mass_valid = self.game.getLegalMoves(state)
        return np.random.choice(len(mass_valid), p=mass_valid / np.sum(mass_valid))


class DeterministicPlayer(Player):
    name: str = "Deterministic"

    def act(self, state: GameState) -> int:
        mass_valid = self.game.getLegalMoves(state)
        indices = np.ravel(np.where(mass_valid == 1))
        return indices[0]


class ManualPlayer(Player):

    def __init__(self, game, name):
        super().__init__(game)
        self.name = name

    def act(self, state: GameState) -> int:
        mass_valid = self.game.getLegalMoves(state)
        indices = np.where(mass_valid == 1)

        move = None
        while move is None:
            move_str = input("Input an integer indicating a move:")
            if move_str.isdigit():
                if int(move_str) in indices:
                    move = int(move_str)

        return move

