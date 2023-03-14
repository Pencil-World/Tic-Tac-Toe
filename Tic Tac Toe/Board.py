import numpy as np

class Board():
    def __init__(self, board = None):
        self.board = np.full([9], 0)
        if board:
            for i, elem in enumerate(board):
                self.board[i] = elem
        self.evaluate()

    def __str__(self):
        board = np.full([3, 3], "")
        dictionary = [" ", "X", "O"]
        for i, elem in enumerate(self.board):
            board[i // 3][i % 3] = dictionary[elem]
        return str(board)

    def __eq__(self, other): 
        return (self.board == other.board).all()

    def __hash__(self):
        val = 0
        for i, elem in enumerate(self.board):
            val += elem * 3 ** i
        return int(val)

    def move(self, action, player):
        self.board[action] = player
        self.evaluate()

    def evaluate(self):
        for (coord, change) in zip([0, 0, 0, 1, 2, 2, 3, 6], [1, 3, 4, 3, 2, 3, 1, 1]):
            if 0 != self.board[coord] == self.board[coord + change] == self.board[coord + 2 * change]:
                self.reward = 10
                return
        self.reward = 1

    def generate(self):
        actions = []
        for i, elem in enumerate(self.board):
            if not elem:
                actions.append(i)
        return np.array(actions)