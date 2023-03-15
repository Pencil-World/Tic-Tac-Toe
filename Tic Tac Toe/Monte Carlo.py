from Board import Board
import datetime
import json
from tensorflow import keras
import numpy as np
import random

def Clear():
    open('log.txt', 'w').close()
    open('buffer.json', 'w').close()

def Load(fstream):
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Loading Data\n")
    global epoch, i, lim, QTable
    QTable = json.load(open(fstream, 'r'))

    f = open('log.txt', 'r')
    log = f.read().split()
    f.close()

    # fix later
    #index = -log[::-1].index("epoch:")
    #epoch = int(log[index])
    #i = lim = (epoch - 1) * cluster_size % episodes

    f = open('log.txt', 'w')
    f.write(''.join([elem + ('\n' if elem.find(':') == -1 else ' ') for elem in log[:index - 1]]))
    f.close()

def Save(fstream):
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Saving Data\n")
        JSON = dict(zip([repr(elem.tolist()) for elem in X], [repr(elem.tolist()) for elem in Y]))
        json.dump(JSON, open(fstream, 'w'), indent = 4)
        json.dump(QTable, open('model.json', 'w'), indent = 4)

        model.save('model.h5')
        debugger.write(f"epoch: {I_AM_ETERNAL * data_size + i}\ntime: {datetime.datetime.now()}\n")

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

alpha = 0.00001
gamma = 0.85
episodes = 1_000
data_size = 1_00

open('debugger.txt', 'w').close()
lim = 100
X = np.zeros([data_size, 3], dtype = np.float32)
Y = np.zeros([data_size, 3], dtype = np.float32)

#Test()
Clear()
#Load('buffer.json')

model = keras.Sequential([
        keras.layers.Dense(125, activation = 'relu',
                            input_shape = [3]),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(25, activation = 'relu'),
        keras.layers.Dense(5, activation = 'relu'),
        keras.layers.Dense(3)])
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0, shuffle = True)
model.summary()

state = Board()
with open('debugger.txt', 'a') as debugger:
    debugger.write("start program\n")
for I_AM_ETERNAL in range(10):
    for i in range(data_size):
        if not (i + 1) % 100:
            Save('buffer.json')
        X[i] = [0, 0, 0]
        Y[i] = model.predict(np.array([[0.9, 0, 0.1]]), verbose = 0) if I_AM_ETERNAL else [random.randrange(0, 100), random.randrange(-100, 0), random.randrange(-50, 50)]
        QTable = dict()
        for epoch in range(lim):
            for temp in range(episodes):
                history = []
                isModel = False
                actions = state.generate()

                while actions.shape[0] and state.reward != 10:
                    if isModel and random.randrange(0, 100) < min(95, epoch * 100 // 25):
                        action = QTable.setdefault(state, np.array([(elem if elem in actions else 0) for elem in range(9)])).argmax()
                    else:
                        action = actions[random.randrange(0, actions.shape[0])]
                    history.append(action)

                    state.move(action, isModel + 1)
                    actions = state.generate()
                    isModel = not isModel

                if state.reward == 10:
                    it = int(isModel)
                else:
                    it = 2
                if lim / 2 <= epoch:
                    X[i][it] += 1
                reward = Y[i][it]

                for action in history[::-1]:
                    isModel = not isModel
                    state.move(action, 0)
                    if isModel:
                        reward = state.reward + gamma * reward
                        mean = QTable.setdefault(state, np.zeros([9]))[action]
                        QTable[state][action] = mean + alpha * (reward - mean)
        X[i] /= episodes * lim / 2

        Qnew = keras.models.clone_model(model)
        Qnew.compile(optimizer = 'adam', loss = 'mse')
        loss = Qnew.fit(X[0:i + 1], Y[0:i + 1], batch_size = 64, epochs = 100, verbose = 0, shuffle = True).history['loss'][-1]
        model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))
        open('log.txt', 'a').write(f"win: {int(X[i][0] * 1000) / 1000}\tloss: {int(X[i][1] * 1000) / 1000}\ttie: {int(X[i][2] * 1000) / 1000}\tloss: {loss}\n")