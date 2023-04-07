import json
import numpy as np

class Board():
    def __init__(self, board = None):
        self.board = np.full([9], 0, dtype = np.int8)
        if board:
            for i, elem in enumerate(board):
                self.board[i] = elem
        self.evaluate()

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        board = np.full([3, 3], "")
        dictionary = [" ", "X", "O"]
        for i, elem in enumerate(self.board):
            board[i // 3][i % 3] = dictionary[elem]
        return str(board)

    def move(self, action, player):
        self.board[action] = player
        self.evaluate()

    def evaluate(self):
        for (coord, change) in zip([0, 0, 0, 1, 2, 2, 3, 6], [1, 3, 4, 3, 2, 3, 1, 1]):
            if 0 != self.board[coord] == self.board[coord + change] == self.board[coord + change + change]:
                self.reward = "RED"
                return
        self.reward = "GREEN"

    def generate(self):
        actions = []
        for i, elem in enumerate(self.board):
            if not elem:
                actions.append(i)
        return np.array(actions, dtype = np.int8)