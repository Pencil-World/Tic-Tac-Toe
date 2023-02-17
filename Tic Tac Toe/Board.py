import numpy as np

class Board():
    def __init__(self, board = None):
        if board:
            self.board = np.zeros([9, 3])
            for i, elem in enumerate(np.array(board).flatten()):
                self.board[i][elem] = 1
        else:
            self.board = np.full([9, 3], [1, 0, 0])
        self.reward = 0

    def __str__(self):
        board = np.full([3, 3], "")
        dictionary = [" ", "X", "O"]
        for i in range(9):
            board[i // 3][i % 3] = dictionary[self.board[i].argmax()]
        return str(board)

    def move(self, action, player):
        self.board[action.argmax()] = player

    def evaluate(self):
        for (coord, change) in zip([0, 0, 0, 1, 2, 2, 3, 6], [1, 3, 4, 3, 2, 3, 1, 1]):
            sum = 0
            for i in range(3):
                sum += all(self.board[coord] == [0, 1, 0])
                coord += change
            
            if sum == 3:
                self.reward = 10
                return
        self.reward = 0

    def generate(self):
        actions = np.zeros([sum([all(elem == [1, 0, 0]) for elem in self.board]), 9])
        i = -1
        temp = self.board.tolist()
        for elem in actions:
            i = temp.index([1, 0, 0], i + 1)
            elem[i] = 1
        return np.array(actions)

    def scrub(self, action):
        return np.concatenate([self.board.flatten(), action.flatten()])
    
    def scrub_all(self, actions):
        arr = np.zeros([actions.shape[0], 36])
        for i, action in enumerate(actions):
            arr[i] = self.scrub(action)
        return arr