from Board import Board
import datetime
import json
from tensorflow import keras
import numpy as np
import random

def Clear():
    open('buffer.json', 'w').close()
    open('debugger.txt', 'w').close()
    open('log.txt', 'w').close()

def Print():
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Saving Data\n")
        debugger.write(f"i: {i}\ntime: {datetime.datetime.now()}\n") # print variables to debugger
        debugger.write(f"win: {HighScore[0]:.3f}\tloss: {HighScore[1]:.3f}\ttie: {HighScore[2]:.3f}\t\t{R}\n")

# edit
def Test():
    print("\nTest")
    QTable = json.load(open('The One.json', 'r'))
    for key, val in QTable.items():
        QTable[key] = np.array(json.loads(val))

    state = Board()
    PlayersTurn = True
    for temp in range(1_000):
        print('\n' + str(state))
        action = int(input()) if PlayersTurn else (QTable[repr(state)].argmax() if random.randrange(0, 100) < 95 and repr(state) in QTable else actions[random.randrange(0, actions.shape[0])])
        state.move(action, 1 if PlayersTurn else 2)

        actions = state.generate()
        PlayersTurn = not PlayersTurn
        if not actions.shape[0] or state.reward == "RED":
            print('\n' + str(state))
            state = Board()
            PlayersTurn = True

alpha = 0.00001
gamma = 0.85
episodes = 1_000
data_size = 1_000

HighScore = [0, 0, 0]
R = [10, -10, 0, 0]
variables = [None] * 2 * 5 # gamma, win reward, lose reward, tie reward, default reward

#open('The One.json', 'w').write(open('model.json').read())
#Clear()
Test()

state = Board()
lim = 10
with open('debugger.txt', 'a') as debugger:
    debugger.write(f"start program\n{datetime.datetime.now()}\n")
for i in range(data_size):
    if not (i + 1) % 100:
        Print()
    val = variables[i // 10 % len(variables)]
    if i // 10 % len(variables):
        print("hi")

    WinLossRatio = [0, 0, 0]
    QTable = dict()
    for epoch in range(lim):
        for temp in range(episodes):
            history = []
            isModel = True
            actions = state.generate()

            while actions.shape[0] and state.reward != "RED":
                mean = None
                action = actions[random.randrange(0, actions.shape[0])]
                if isModel:
                    mean = QTable.setdefault(repr(state), np.full([9], -1000, dtype = np.float32))
                    index = mean.argmax() if random.randrange(0, 100) < epoch * 10 else None
                    action = index if index and mean[index] != -1000 else action
                history.append((action, mean))
                state.move(action, isModel + 1)
                actions = state.generate()
                isModel = not isModel

            it = int(isModel) if state.reward == "RED" else 2
            WinLossRatio[it] += epoch + 1 == lim
            reward = R[it]

            prev = None
            for action, mean in history[::-1]:
                isModel = not isModel
                state.move(action, 0)
                if isModel:
                    reward = R[3] + gamma * reward
                    mean[action][prev] = reward if mean[action] == -1000 else mean[action] + alpha * (reward - mean[action]) # if prev is none, set all possibilities of prev to the same value (as if opponent does not play a move)
                else:
                    prev = action

    WinLossRatio /= episodes
    if HighScore[0] < WinLossRatio[0]:
        with open('debugger.txt', 'a') as debugger:
            debugger.write(f"{HighScore[0]:.3f}-{WinLossRatio[0]:.3f}\tgamma: {gamma}\twin reward: {R[0]}\tloss reward: {R[1]}\ttie reward: {R[2]}\n")
        if HighScore[0] < WinLossRatio[0] or (HighScore[0] == WinLossRatio[0] and HighScore[1] < WinLossRatio[1]):
            HighScore = WinLossRatio
            JSON = dict(zip(QTable.keys(), [repr(elem.tolist()) for elem in QTable.values()]))
            json.dump(JSON, open('model.json', 'w'), indent = 4)

    open('log.txt', 'a').write(f"win rate: {WinLossRatio[0]:.3f}\tloss rate: {WinLossRatio[1]:.3f}\ttie rate: {WinLossRatio[2]:.3f}\n")