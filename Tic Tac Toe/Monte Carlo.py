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
        debugger.write(f"time: {datetime.datetime.now()}\ti: {i}\tbounds: {str(bounds.tolist())}")

#def Test():
#    print("\nTest")
#    QTable = json.load(open('The One.json', 'r'))
#    for key, val in QTable.items():
#        QTable[key] = np.array(json.loads(val))

#    state = Board()
#    PlayersTurn = True
#    for temp in range(1_000):
#        print('\n' + str(state))
#        action = int(input()) if PlayersTurn else (QTable[repr(state)].argmax() if random.randrange(0, 100) < 95 and repr(state) in QTable else actions[random.randrange(0, actions.shape[0])])
#        state.move(action, 1 if PlayersTurn else 2)

#        actions = state.generate()
#        PlayersTurn = not PlayersTurn
#        if not actions.shape[0] or state.reward == "RED":
#            print('\n' + str(state))
#            state = Board()
#            PlayersTurn = True

alpha = 0.00001
gamma = 0.5
episodes = 1_000
data_size = 1_000

# polynomial regression always converges to a global minimum with MSE
HighScore = np.zeros([3])
R = [10, -10, 0, 0] # win reward, lose reward, tie reward, default reward
bounds = np.zeros([1, 2, 2]) # includes gamma as the 0th element

#open('The One.json', 'w').write(open('model.json').read())
Clear()
#Test()

state = Board()
lim = 10
with open('debugger.txt', 'a') as debugger:
    debugger.write(f"start program\n{datetime.datetime.now()}\n")
for i in range(data_size):
    if not (i + 1) % 100:
        Print()
    it = i / 10 % bounds.shape[0]
    if it < 0.2:
        gamma = 0.0001 if it == 0 else 0.9999
    elif int(it) == 0:
        gamma = max(0, min((bounds[0][0][0] + bounds[0][1][0]) / 2, 1))
    #elif it % 1 < 0.2:
    #    R[int(it) - 1] = 100 * (-1 if it % 1 else 1)
    #else:
    #    R[int(it) - 1] = (bounds[int(it)][0][0] + bounds[int(it)][1][0]) / 2

    it = int(it)
    CurrScore = np.zeros([3])
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
                    index = mean.argmax() if random.randrange(0, 100) < epoch * (100 // lim) else None
                    action = index if index and mean[index] != -1000 else action
                history.append((action, mean))
                state.move(action, isModel + 1)
                actions = state.generate()
                isModel = not isModel

            it = int(isModel) if state.reward == "RED" else 2
            CurrScore[it] += epoch + 1 == lim
            reward = R[it]

            for action, mean in history[::-1]:
                isModel = not isModel
                state.move(action, 0)
                if isModel:
                    reward = R[3] + gamma * reward
                    mean[action] = reward if mean[action] == -1000 else mean[action] + alpha * (reward - mean[action])

    CurrScore /= episodes
    open('log.txt', 'a').write(f"win rate: {CurrScore[0]:.3f}\tloss rate: {CurrScore[1]:.3f}\ttie rate: {CurrScore[2]:.3f}\n")
    if HighScore[0] < CurrScore[0]:
        with open('debugger.txt', 'a') as debugger:
            debugger.write(f"{HighScore[0]:.3f}-{CurrScore[0]:.3f}\tgamma: {gamma}\twin reward: {R[0]}\tloss reward: {R[1]}\ttie reward: {R[2]}\tdefault reward: {R[3]}\n")
        HighScore = CurrScore
        JSON = dict(zip(QTable.keys(), [repr(elem.tolist()) for elem in QTable.values()])) # works for jagged arrays. includes commas
        json.dump(JSON, open('model.json', 'w'), indent = 4)

        bounds[it][bounds[it][:,1].argmin()] = np.array([gamma if it == 0 else R[it - 1], CurrScore[0]])