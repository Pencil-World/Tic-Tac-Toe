from Board import Board
import datetime
import json
from tensorflow import keras
import numpy as np
import random
import time

# def Clear():
#     open('log.txt', 'w').close()
#     open('buffer.json', 'w').close()

#def Load(fstream):
#    with open('debugger.txt', 'a') as debugger:
#        debugger.write("Loading Data\n")
#    global epoch, i, lim, QTable
#    QTable = json.load(open(fstream, 'r'))

#    f = open('log.txt', 'r')
#    log = f.read().split()
#    f.close()

#    index = -log[::-1].index("epoch:")
#    epoch = int(log[index])
#    i = lim = (epoch - 1) * cluster_size % episodes

#    f = open('log.txt', 'w')
#    f.write(''.join([elem + ('\n' if elem.find(':') == -1 else ' ') for elem in log[:index - 1]]))
#    f.close()

#def Save(fstream):
#    with open('debugger.txt', 'a') as debugger:
#        #debugger.write("Saving Data\n")
#        json.dump(QTable, open(fstream, 'w'), indent = 4)
#        text = f"epoch: {epoch}\ntime: {time.time() - Time}\n"
#        open('log.txt', 'a').write(text)
#        #debugger.write(text)

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
lim = 300

alpha = 0.00001
gamma = 0.85
episodes = 1_000
data_size = 1_000

X = np.zeros([data_size, 3], dtype = np.float32)
Y = np.zeros([data_size, 3], dtype = np.float32)

#Test()
#Clear()
#Load('buffer.json')

model = keras.Sequential([
        keras.layers.Dense(125, activation = 'relu',
                            input_shape = [3]),
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
        Y[i] = model.predict(np.array([[0.8, 0, 0.2]]), verbose = 0)
        QTable = dict()
        for epoch in range(lim):
            #Save('buffer.json')

            WinLossRatio = [0, 0, 0]
            for temp in range(episodes):
                history = []
                isModel = False
                actions = state.generate()

                while True:
                    if isModel and random.randrange(0, 100) < min(95, epoch * 100 // 25) and state in QTable:
                        action = QTable[state].argmax()
                    else:
                        action = actions[random.randrange(0, actions.shape[0])]
                    history.append(action)

                    state.move(action, isModel + 1)
                    actions = state.generate()

                    if not actions.shape[0] or state.reward == 10:
                        if state.reward == 10:
                            it = int(not isModel)
                        else:
                            it = 2
                        WinLossRatio[it] += 1
                        reward = Y[i][it]
                        break
                    isModel = not isModel

                for action in history[::-1]:
                    state.move(action, 0)
                    if isModel:
                        reward = state.reward + gamma * reward#bigg issueee??? what should default state.reward equal!!!
                        if state in QTable:
                            mean = QTable[state][action]
                        else:
                            QTable[state] = np.zeros([9])
                            mean = reward
                        QTable[state][action] = mean + alpha * (reward - mean)
                    isModel = not isModel

            if lim / 2 <= epoch:
                X[i] += WinLossRatio
            with open('debugger.txt', 'a') as debugger:
                debugger.write(f"win to loss to tie ratio:\t{WinLossRatio[0] / episodes}\t{WinLossRatio[1] / episodes}\t{WinLossRatio[2] / episodes}\n")
        X[i] /= episodes * lim / 2

        Qnew = keras.models.clone_model(model)
        Qnew.compile(optimizer = 'adam', loss = 'mse')
        loss = Qnew.fit(X[0:i + 1], Y[0:i + 1], batch_size = 64, epochs = 100, verbose = 0, shuffle = True).history['loss'][-1]
        model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))
        print(loss)