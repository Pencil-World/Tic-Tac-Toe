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

    # fix later. get index, epoch, and i from length of array. how many lines it is and truncate/floor
    #index = -log[::-1].index("epoch:")
    #epoch = int(log[index])
    #i = lim = (epoch - 1) * cluster_size % episodes
    model.compile(optimizer = 'adam', loss = 'mse')
    model.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0, shuffle = True)

    f = open('log.txt', 'w')
    f.write(''.join([elem + ('\n' if elem.find(':') == -1 else ' ') for elem in log[:index - 1]]))
    f.close()

def Save(fstream):
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Saving Data\n")

        JSON = dict(zip([str(elem) for elem in X], [str(elem) for elem in Y]))
        json.dump(JSON, open(fstream, 'w'), indent = 4)
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
data_size = 1_000

open('debugger.txt', 'w').close()
lim = 100
X = np.zeros([data_size, 3], dtype = np.float32)
Y = np.zeros([data_size, 3], dtype = np.float32)

print(repr(Board()))
print(Board())

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
model.summary()

state = Board()
HighScore = 0
with open('debugger.txt', 'a') as debugger:
    debugger.write(f"start program\n{datetime.datetime.now()}\n")
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
                    mean = None
                    action = actions[random.randrange(0, actions.shape[0])]
                    if isModel:
                        mean = QTable.setdefault(repr(state), np.zeros([9]))
                        action = mean.argmax() if random.randrange(0, 100) < min(95, epoch * 100 // 25) and mean.max() else action
                    history.append((action, mean))
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

                for action, mean in history[::-1]:
                    isModel = not isModel
                    state.move(action, 0)
                    if isModel:
                        reward = state.reward + gamma * reward
                        mean[action] = mean[action] + alpha * (reward - mean[action])

        X[i] /= episodes * lim / 2
        if 0.975 * HighScore < X[i][0]:
            HighScore = max(HighScore, X[i][0])
            JSON = dict(zip(QTable.keys(), [str(elem) for elem in QTable.values()]))
            json.dump(JSON, open('model.json', 'w'), indent = 4)
            with open('debugger.txt', 'a') as debugger:
                debugger.write(f"{int(HighScore * 1000) / 1000} - {int(X[i][0] * 1000) / 1000}\twin: {int(Y[i][0] * 1000) / 1000}\tloss: {int(Y[i][1] * 1000) / 1000}\ttie: {int(Y[i][2] * 1000) / 1000}\n")

        Qnew = keras.models.clone_model(model)
        Qnew.compile(optimizer = 'adam', loss = 'mse')
        loss = Qnew.fit(X[:None if I_AM_ETERNAL else i + 1], Y[:None if I_AM_ETERNAL else i + 1], batch_size = 64, epochs = 100, verbose = 0, shuffle = True).history['loss'][-1]
        model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))
        open('log.txt', 'a').write(f"win: {int(X[i][0] * 1000) / 1000}\tloss: {int(X[i][1] * 1000) / 1000}\ttie: {int(X[i][2] * 1000) / 1000}\tloss: {loss}\n")