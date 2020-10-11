import numpy as np

from utils.selfplay_utils import GameHistory


class Player:

    def __init__(self, game) -> None:
        self.game = game
        self.history = None

    def bind_history(self, history: GameHistory):
        self.history = history

    def act(self, state: np.ndarray, player: int) -> int:
        pass


class AlphaZeroPlayer(Player):

    def __init__(self, game, search_engine, model) -> None:
        super().__init__(game)
        self.search_engine = search_engine
        self.model = model

    def act(self, state: np.ndarray, player: int) -> int:
        pi = self.search_engine.getActionProb(state, temp=0)
        return np.argmax(pi).item()


class MuZeroPlayer(Player):

    def __init__(self, game, search_engine, model) -> None:
        super().__init__(game)
        self.search_engine = search_engine
        self.model = model

    def act(self, state: np.ndarray, player: int) -> int:
        o_t = self.game.buildObservation(state, player, form=self.game.Observation.HEURISTIC)
        stacked_observations = self.history.stackObservations(self.model.net_args.observation_length, o_t)

        root_actions = self.game.getLegalMoves(state, player)
        pi, _ = self.search_engine.runMCTS(stacked_observations, legal_moves=root_actions, temp=0)

        return np.argmax(pi).item()


class RandomPlayer(Player):

    def act(self, state, player):
        mass_valid = self.game.getLegalMoves(state, player)
        return np.random.choice(len(mass_valid), p=mass_valid / np.sum(mass_valid))


class DeterministicPlayer(Player):

    def act(self, state: np.ndarray, player: int) -> int:
        mass_valid = self.game.getLegalMoves(state, player)
        indices = np.where(mass_valid == 1)
        return indices[0]


class ManualPlayer(Player):

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

