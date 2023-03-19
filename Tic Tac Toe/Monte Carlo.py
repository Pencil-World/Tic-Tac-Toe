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

def Load():
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Loading Data\n")
    global model, i, I_AM_ETERNAL
    model = keras.models.load_model('model.h5')

    log = open('debugger.txt', 'r').read().split()
    epoch = int(log[-log[::-1].index("epoch:")])
    I_AM_ETERNAL = epoch // data_size
    i = epoch % data_size

    JSON = json.load(open('buffer.json', 'r'))
    length = len(JSON.items())
    X[:length] = [np.array(json.loads(elem)) for elem in JSON.keys()]
    Y[:length] = [np.array(json.loads(elem)) for elem in JSON.values()]

    for it, _i in enumerate(range(length, data_size)):
        X[_i] = X[it]
        Y[_i] = Y[it]

def Save():
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Saving Data\n")

        JSON = dict(zip([repr(elem.tolist()) for elem in X], [repr(elem.tolist()) for elem in Y]))
        json.dump(JSON, open('buffer.json', 'w'), indent = 4)
        model.save('model.h5')
        debugger.write(f"epoch: {I_AM_ETERNAL * data_size + i}\ntime: {datetime.datetime.now()}\n")
        debugger.write(f"win: {HighScore[0]:.3f}\tloss: {HighScore[1]:.3f}\ttie: {HighScore[2]:.3f}\t\t{model(np.array([HighScore]))}\n")

def Test():
    print("\nTest")
    QTable = json.load(open('model.json', 'r'))
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
        if not actions.shape[0] or state.reward == 10:
            print('\n' + str(state))
            state = Board()
            PlayersTurn = True

alpha = 0.00001
gamma = 0.85
episodes = 1_000
data_size = 1_000

lim = 100
I_AM_ETERNAL = i = 0
X = np.zeros([data_size, 3], dtype = np.float32)
Y = np.zeros([data_size, 3], dtype = np.float32)

model = keras.Sequential([
        keras.layers.Dense(125, activation = 'relu',
                            input_shape = [3]),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(25, activation = 'relu'),
        keras.layers.Dense(3)])
model.compile(optimizer = 'adam', loss = 'mse')
model.summary()

#open('The One.json', 'w').write(open('model.json').read())
#Clear()
#Load()
#Test()

state = Board()
HighScore = [0, 0, 0]
with open('debugger.txt', 'a') as debugger:
    debugger.write(f"start program\n{datetime.datetime.now()}\n")
for I_AM_ETERNAL in range(I_AM_ETERNAL, 10):
    for i in range(i, data_size):
        if not (i + 1) % 100:
            Save()
        X[i] = [0, 0, 0]
        Y[i] = model(np.array([HighScore])) if I_AM_ETERNAL else [random.randint(0, 100), random.randint(-100, 0), random.randint(-50, 50)]
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
                        mean = QTable.setdefault(repr(state), np.zeros([9], dtype = np.float32))
                        action = mean.argmax() if random.randrange(0, 100) < min(95, epoch * 100 // 25) and np.count_nonzero(mean) else action
                    history.append((action, mean))
                    state.move(action, isModel + 1)
                    actions = state.generate()
                    isModel = not isModel

                it = int(isModel) if state.reward == 10 else 2
                X[i][it] += lim / 2 <= epoch
                reward = Y[i][it]

                for action, mean in history[::-1]:
                    isModel = not isModel
                    state.move(action, 0)
                    if isModel:
                        reward = state.reward + gamma * reward
                        mean[action] = mean[action] + alpha * (reward - mean[action])

        X[i] /= episodes * lim / 2
        if 0.99 * HighScore[0] < X[i][0]:
            with open('debugger.txt', 'a') as debugger:
                debugger.write(f"{HighScore[0]:.3f}-{X[i][0]:.3f}\twin: {Y[i][0]}\tloss: {Y[i][1]}\ttie: {Y[i][2]}\n")
            if HighScore[0] < X[i][0] or (HighScore[0] == X[i][0] and HighScore[1] < X[i][0]):
                HighScore = X[i]
                JSON = dict(zip(QTable.keys(), [repr(elem.tolist()) for elem in QTable.values()]))
                json.dump(JSON, open('model.json', 'w'), indent = 4)

        if I_AM_ETERNAL:
            Qnew = keras.models.clone_model(model)
            Qnew.compile(optimizer = 'adam', loss = 'mse')
            loss = Qnew.fit(X, Y, epochs = 100, verbose = 0).history['loss'][-1]
            model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))
        else:
            loss = model.fit(X[:i + 1], Y[:i + 1], epochs = 100, verbose = 0).history['loss'][-1]
        open('log.txt', 'a').write(f"win: {X[i][0]:.3f}\tloss: {X[i][1]:.3f}\ttie: {X[i][2]:.3f}\tloss: {loss:.3f}\n")
    i = 0