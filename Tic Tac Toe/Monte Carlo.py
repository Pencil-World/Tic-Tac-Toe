from Board import Board
import datetime
import json
import numpy as np
import random

def Clear():
    open('buffer.json', 'w').close()
    open('debugger.txt', 'w').close()
    open('log.txt', 'w').close()

def Experiment():
    global gamma
    P = random.gauss(0.5, 0.25)
    pi = i / 10 % bounds.shape[0]
    if pi < 0.2:
        gamma = 0.00001 if pi == 0 else 0.99999
        if pi == 0:
            Print()
            bounds[int(pi)] = np.zeros([2, 2])
    elif int(pi) == 0:
        gamma = max(0, min(P * bounds[0][0][0] + (1 - P) * bounds[0][1][0], 1))
    #elif pi % 1 < 0.2:
    #    R[int(pi) - 1] = 100 * (-1 if pi % 1 else 1)
    #else:
    #    R[int(pi) - 1] = max(-999, P * bounds[int(pi)][0][0] + (1 - P) * bounds[int(pi)][1][0])

    pi = int(pi)
    return [bounds[pi][bounds[pi][:,1].argmin():bounds[pi][:,1].argmin() + 1], gamma if pi == 0 else R[pi - 1]]

def NewRecord():
    global HighScore
    with open('debugger.txt', 'a') as debugger:
        debugger.write(f"{HighScore[0]:.3f}-{CurrScore[0]:.3f}\tgamma: {gamma}\twin reward: {R[0]}\tloss reward: {R[1]}\ttie reward: {R[2]}\tdefault reward: {R[3]}\n")
    HighScore = CurrScore
    JSON = dict(zip(QTable.keys(), [repr(elem.tolist()) for elem in QTable.values()])) # works for jagged arrays. includes commas
    json.dump(JSON, open('model.json', 'w'), indent = 4)

def Print():
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Saving Data\n")
        debugger.write(f"time: {datetime.datetime.now()}\ti: {i}\tbounds: {str(bounds.tolist())}\n")

def Test():
    print("\nTest")
    QTable = json.load(open('The One.json', 'r'))
    for key, val in QTable.items():
        QTable[key] = np.array(json.loads(val))

    state = Board()
    ModelTurn = True
    while True:
        action = int(input()) if not ModelTurn else (QTable[repr(state)].argmax() if random.randrange(0, 100) < 95 and repr(state) in QTable else actions[random.randrange(0, actions.shape[0])])
        state.move(action, ModelTurn + 1)
        print('\n' + str(state))

        actions = state.generate()
        ModelTurn = not ModelTurn
        if not actions.shape[0] or state.progress == "RED":
            print('\n' + str(state))
            state = Board()
            ModelTurn = True

alpha = 0.00001
gamma = 0.5
episodes = 1_000
data_size = 1_000

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
    target = Experiment()
    CurrScore = np.zeros([3])
    QTable = dict()
    for epoch in range(lim):
        for temp in range(episodes):
            history = []
            ModelTurn = True
            actions = state.generate()

            # forward propagation
            while actions.shape[0] and state.progress != "RED":
                mean = None
                action = actions[random.randrange(0, actions.shape[0])]
                if ModelTurn:
                    mean = QTable.setdefault(repr(state), np.full([9], -1000, dtype = np.float32))
                    index = mean.argmax() if random.randrange(0, 100) < epoch * (100 // lim) else None
                    action = index if index and mean[index] != -1000 else action
                history.append((action, mean))
                state.move(action, ModelTurn + 1)
                actions = state.generate()
                ModelTurn = not ModelTurn

            it = int(ModelTurn) if state.progress == "RED" else 2
            reward = R[it]
            CurrScore[it] += epoch + 1 == lim

            # backward propagation
            for action, mean in history[::-1]:
                ModelTurn = not ModelTurn
                state.move(action, 0)
                if ModelTurn:
                    reward = R[3] + gamma * reward
                    mean[action] = reward if mean[action] == -1000 else mean[action] + alpha * (reward - mean[action])

    CurrScore /= episodes
    open('log.txt', 'a').write(f"win rate: {CurrScore[0]:.3f}\tloss rate: {CurrScore[1]:.3f}\ttie rate: {CurrScore[2]:.3f}\n")
    if target[0][0][1] < CurrScore[0]: # [1] retreives the win rate of the lowest bound. [0] retreives the value of the lowest bound
        target[0][0] = [target[1], CurrScore[0]] # replaces the lowest bound with a higher win rate value. [0] captures the slice by reference
    if HighScore[0] < CurrScore[0]:
        NewRecord()