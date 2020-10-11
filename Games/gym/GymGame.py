from Game import Game


class GymGame(Game):

    def __init__(self, env_name):
        super().__init__(n_players=1)
