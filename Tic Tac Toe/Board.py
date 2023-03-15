import json
import numpy as np

class Board():
    def __init__(self, board = None, dict = None):
        self.board = np.full([9], 0)
        if board:
            for i, elem in enumerate(board):
                self.board[i] = elem
        elif dict:
            for key, val in dict.items():
                if type(val) is list:
                    val = np.array(val)
                setattr(self, key, val)
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

    @staticmethod
    def json_dumps(object):
        return json.dumps(object, default = lambda o: (o.tolist() if type(o) is np.ndarray else o.__dict__), 
            sort_keys = True, indent = 4)

    @staticmethod
    def json_loads(object):
        return Board(dict = json.loads(object))

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