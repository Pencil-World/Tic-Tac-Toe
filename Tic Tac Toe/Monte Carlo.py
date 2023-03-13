from Board import Board
import datetime
import json
from tensorflow import keras
import numpy as np
import random
import time

def Clear():
    open('log.txt', 'w').close()
    open('buffer.json', 'w').close()

def Load(fstream):
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Loading Data\n")
    global epoch, i, lim, model

    data = json.load(open(fstream, 'r'))
    length = len(data.items())
    X[:length] = [np.array(json.loads(elem)) for elem in data.keys()]
    Y[:length] = list(data.values())

    for it, _i in enumerate(range(length, data_size)):
        X[_i] = X[it]
        Y[_i] = Y[it]

    f = open('log.txt', 'r')
    log = f.read().split()
    f.close()

    model = keras.models.load_model('model.h5')
    index = -log[::-1].index("epoch:")
    epoch = int(log[index])
    i = lim = (epoch - 1) * cluster_size % data_size

    f = open('log.txt', 'w')
    f.write(''.join([elem + ('\n' if elem.find(':') == -1 else ' ') for elem in log[:index - 1]]))
    f.close()

def Save(fstream):
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Saving Data\n")
   
        JSON = dict(zip([repr(elem.tolist()) for elem in X], Y))
        json.dump(JSON, open(fstream, 'w'), indent = 4)

        model.save('model.h5')
        text = f"epoch: {epoch}\ntime: {time.time() - Time}\n"
        open('log.txt', 'a').write(text)
        debugger.write(text)

def Test():
    print("\nTest")
    model = keras.models.load_model('model.h5')

    table = np.zeros([9, 9])
    for i, elem in enumerate(table):
        elem[i] = 1

    state = Board()
    PlayersTurn = True
    for temp in range(1_000):
        print()
        print(state)

        actions = state.generate()
        if PlayersTurn:
            action = table[int(input())]
            state.move(action, [0, 1, 0])
        else:
            value = model.predict(state.scrub_all(actions), verbose = 0)
            action = actions[value.argmax() if random.randrange(0, 100) < 95 else random.randrange(0, actions.shape[0])]
            state.move(action, [0, 0, 1])

        PlayersTurn = not PlayersTurn
        if not state.generate().shape[0] or state.reward == 10:
            print()
            print(state)
            state = Board()
            PlayersTurn = True

open('debugger.txt', 'w').close()
Time = time.time()
epoch = 1
i = lim = 0

alpha = 1 / 100
discount = 0.85
data_size = 10_000
cluster_size = 1_000

QTable = dict()
#Synthesize()
#Test()
Load('data.json')
Clear()
#Load('buffer.json')

state = Board()
with open('debugger.txt', 'a') as debugger:
    debugger.write("start program\n")
for epoch in range(epoch, 1_000):
    #Save('buffer.json')

    if i == data_size:
        i = lim = 0
    lim += cluster_size
    WinLossRatio = [0, 0]
    while i < lim:
        # simulate environment
        history = []
        isModel = False
        actions = state.generate()

        for temp in range(2 * (data_size - i)):
            action = actions[QTable[state].argmax() if isModel and random.randrange(0, 100) < min(95, epoch * 100 // 25) else random.randrange(0, actions.shape[0])]
            history.append(action)

            state.move(action, [0, 0, 1] if isModel else [0, 1, 0])
            actions = state.generate()

            if not actions.shape[0] or state.reward == 10:
                if state.reward == 10:
                    WinLossRatio[not isModel] += 1
                break
            isModel = not isModel

        # replay buffer
        reward = 10 if isModel else 1
        for action in history[::-1]:
            state.move(action, [1, 0, 0])

            if isModel:
                reward = state.reward + discount * reward
                mean = QTable[state][action]
                QTable[state][action] = mean + alpha * (reward - mean)

            # train model
            i += isModel
            if isModel and not i % 100:
                with open('debugger.txt', 'a') as debugger:
                    debugger.write(f"{datetime.datetime.now()} " + text)
            isModel = not isModel

    with open('debugger.txt', 'a') as debugger:
        debugger.write(f"win to loss ratio (expected to be between 50% and 90%): {WinLossRatio[0] * 100 / sum(WinLossRatio)} percent\n")