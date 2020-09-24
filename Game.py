class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    single-player, two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2 (if |players| > 1).

    See hex/HexGame.py for an example implementation.

    Call the necessary super() methods to be able to utilize the GameHistory.
    """

    class GameHistory:

        def __init__(self):
            self.state_history = list()
            self.player_history = list()
            self.action_history = list()

        def capture(self, state, player, action):
            self.state_history.append(state)
            self.player_history.append(player)
            self.action_history.append(action)

        def refresh(self):
            self.state_history = list()
            self.player_history = list()
            self.action_history = list()

    def __init__(self):
        self.history = Game.GameHistory()

    def getInitBoard(self):
        """
        Returns:
            startState: a representation of the initial state (ideally this is the form
                        that will be the input to your neural network)
        """
        self.history.refresh()

    def getDimensions(self):
        """
        Returns:
            (x,y): a tuple of the state dimensions
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def getNextState(self, state, player, action):
        """
        Input:
            state: current state
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn
        """
        self.history.capture(state, player, action)

    def getLegalMoves(self, state, player):
        """
        Input:
            board: current state
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are legal 0 for invalid moves
        """
        pass

    def getGameEnded(self, state, player):
        """
        Input:
            state: current state
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

    def getCanonicalForm(self, state, player):
        """
        Input:
            board: current state
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of the state. The canonical form
                            should be independent of the player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, state, pi):
        """
        Input:
            state: current state
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(state,pi)] where each tuple is a symmetrical
                       form of the state and the corresponding pi vector. This
                       is used when training AlphaZero from examples.
        """
        pass

    def stringRepresentation(self, state):
        """
        Input:
            state: current state

        Returns:
            stateString: a quick conversion of state to a string format.
                         Required by MCTS for hashing.
        """
        pass
