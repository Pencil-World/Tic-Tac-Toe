from Board import Board
import copy
import json
from tensorflow import keras
import numpy as np
import random
import time

def Load(fstream):
    print("\nLoading Data")

    data = json.load(open(fstream, 'r'))
    for _i, (key, val) in enumerate(data.items()):
        X[_i] = np.array(json.loads(key))
        Y[_i] = val
    for _i in range(_i + 1, 10_000):
        X[_i] = X[_i - 5_000]
        Y[_i] = Y[_i - 5_000]

    if fstream == 'data.json':
        return

    global epoch, i, model
    model = keras.models.load_model('model.h5')
    f = open('log.txt', 'w+')
    log = f.read().split()

    index = -log[::-1].index("epoch:")
    epoch = int(log[index])
    i = (epoch - 1) * cluster_size % data_size
    f.write(''.join([elem + ('\n' if elem.find(':') == -1 else ' ') for elem in log[:index - 1]]))

    for i, elem in enumerate(log[::2]):
        if elem == "loss:":
            loss = log[i * 2 + 1]
            print(f"loss: {'x' * min(100, int(loss // 4))}")

def Save(fstream):
    print("\nSaving Data")
   
    JSON = dict()
    for i in range(data_size):
        JSON[np.array2string(X[i], separator = ", ", max_line_width = 1_000)] = float(Y[i])
    json.dump(JSON, open(fstream, 'w'), indent = 4)

    if fstream == 'data.json':
        return

    model.save('model.h5')
    text = f"epoch: {epoch} time: {time.time() - Time}"
    open('log.txt', 'a').write(text + '\n')
    print(text)

def Test():
    print("\nTest")
    model = keras.models.load_model('model.h5')

    state = Board()
    other = Board()
    table = np.zeros([9, 9])
    for i, elem in enumerate(table):
        elem[i] = 1

    for i in range(1_000):
        print()
        print(state if i % 2 else other)

        actions = state.generate()
        value = model.predict(state.scrub_all(actions), verbose = 0)
        action = actions[value.argmax() if random.randrange(0, 100) < 95 else random.randrange(0, actions.shape[0])]
        if i % 2:
            action = table[int(input())]

        other.move(action, [0, 0, 1])
        state.move(action, [0, 1, 0])

        if not state.generate().shape[0] or state.reward == 10:
            print()
            print(state if i % 2 else other)
            state = Board()
            other = Board()
        state, other = other, state

def Clear():
    open('log.txt', 'w').close()
    open('buffer.json', 'w').close()

def Synthesize():
    print("\nSynthesizing Data")
    for epoch in range(1, 11):
        i = 0
        while i < data_size * epoch // 10:
            history = []
            state, other = Board(), Board()
            actions = state.generate()
            while actions.shape[0] and state.reward != 10:
                state, other = other, state
                action = actions[random.randrange(0, actions.shape[0])]
                history.append(action)

                other.move(action, [0, 0, 1])
                state.move(action, [0, 1, 0])
                actions = state.generate()
                if epoch == 10:# swap before break
                    state, other = other, state
                    break
            else:
                for elem in history[:-1 - epoch:-1]:
                    other.move(elem, [1, 0, 0])
                    state.move(elem, [1, 0, 0])
                actions = state.generate()
            
            isModel = True
            value = model.predict(state.scrub_all(actions), verbose = 0)
            for i in range(epoch):
                action = actions[random.randrange(0, actions.shape[0])]
                if isModel:
                    reward = state.reward
                    X[i] = state.scrub(action)

                other.move(action, [0, 0, 1])
                state.move(action, [0, 1, 0])
                actions = state.generate()
    
                if not actions.shape[0] or state.reward == 10:
                    Y[i] = reward + discount * (state.reward if isModel else other.reward)
                    i += 1
                    break
                else:
                    state, other = other, state
                    isModel = not isModel
                    value = (model if isModel else paragon).predict(state.scrub_all(actions), verbose = 0)
                    if model:
                        if actions.shape[0] < 8:
                            Y[i] = reward + discount * value.max()
                        i += 1

    Save('data.json')

epoch = 1
i = lim = 0
discount = 0.85
data_size = 10_000
cluster_size = 1_000
shape = Board().scrub_all(Board().generate()).shape[1]
X = np.zeros([data_size, shape], dtype = np.int8)
Y = np.zeros([data_size], dtype = np.float32)

#Test()
#Load('data.json')
Clear()

model = keras.Sequential([
        keras.layers.Dense(125, activation = 'relu',
                            input_shape = [shape]),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(25, activation = 'relu'),
        keras.layers.Dense(5, activation = 'relu'),
        keras.layers.Dense(1)])
model.compile(optimizer = 'adam', loss = 'mse')
model.summary()
model.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0)
Time = time.time()
sentries = [model] * 100

#Load('buffer.json')
Synthesize()
print("start program")
for epoch in range(epoch, 1_000):
    Save('buffer.json')

    if i == data_size:
        i = lim = 0
    lim += cluster_size
    while i < lim:
        # simulate environment
        state, other = Board(), Board()
        paragon = sentries[random.randrange(0, min(epoch, 100))]
        isModel = True
        if random.randint(0, 1):
            actions = state.generate()
            action = actions[random.randrange(0, actions.shape[0])]
            other.move(action, [0, 1, 0])
            state.move(action, [0, 0, 1])

        # replay buffer
        actions = state.generate()
        value = model.predict(state.scrub_all(actions), verbose = 0)
        for temp in range(min(epoch, data_size - i)):
            action = actions[value.argmax() if random.randrange(0, 100) < 9 else random.randrange(0, actions.shape[0])]
            if isModel:
                reward = state.reward
                X[i] = state.scrub(action)

            other.move(action, [0, 0, 1])
            state.move(action, [0, 1, 0])
            actions = state.generate()

            if not actions.shape[0] or state.reward == 10:
                Y[i] = reward + discount * (state.reward if isModel else other.reward)
                i += 1
            else:
                state, other = other, state
                isModel = not isModel
                value = (model if isModel else paragon).predict(state.scrub_all(actions), verbose = 0)
                if model:
                    if actions.shape[0] < 8:
                        Y[i] = reward + discount * value.max()
                    i += 1

            # train model
            if isModel and not i % 100:
                Qnew = keras.models.clone_model(model)
                Qnew.compile(optimizer = 'adam', loss = 'mse')
                loss = Qnew.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0).history['loss'][-1]
                model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))

                text = f"loss: {loss}"
                open('log.txt', 'a').write(text + '\n')
                print(text)

            if not actions.shape[0] or state.reward == 10:
                break

    if epoch < 100:
        sentries[epoch] = keras.models.clone_model(model)
        sentries[epoch].set_weights(model.get_weights())
        sentries[epoch].compile(optimizer = 'adam', loss = 'mse')