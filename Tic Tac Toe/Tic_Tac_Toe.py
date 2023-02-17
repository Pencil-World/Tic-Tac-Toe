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
    for i, (key, val) in enumerate(data.items()):
        X[i] = np.array(json.loads(key))
        Y[i] = val

    if fstream == 'data.json':
        return

    global epoch, model
    model = keras.models.load_model('model.h5')

    # does this work?
    log = open('log.txt', 'r').read().split()
    epoch = int(log[-log[::-1].index("epoch:")])
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
        actions = state.generate()
        value = model.predict(state.scrub_all(actions), verbose = 0)
        action = actions[value.argmax() if 50 * (actions.shape[0] - 7) < random.randrange(0, 100) else random.randrange(0, actions.shape[0])]
        #if i % 2:
        #    action = table[input()]

        other.move(action, [0, 0, 1])
        state.move(action, [0, 1, 0])
        print()
        print(state if i % 2 else other)

        if not state.generate().shape[0] or state.reward == 10:
            state = Board()
            other = Board()
        state, other = other, state

def Clear():
    open('log.txt', 'w').close()
    open('buffer.json', 'w').close()

def Synthesize():
    print("\nSynthesizing Data")
    state, other = Board(), Board()
    i = 0
    while i < 2_500:
        reward = state.reward
        actions = state.generate()
        action = actions[random.randrange(0, actions.shape[0])]
        X[i] = state.scrub(action)

        other.move(action, [0, 0, 1])
        state.move(action, [0, 1, 0])
    
        if not state.generate().shape[0] or state.reward == 10:
            if state.reward == 10:
                if not X[i].tolist() in X[:i].tolist():
                    Y[i] = Y[i + 2_500] = Y[i + 5_000] = Y[i + 7_500] = reward + discount * state.reward
                    i += 1

            state = Board()
            other = Board()
        else:
            state, other = other, state
    
    Save('data.json')

discount = 0.85
data_size = 10_000
shape = Board().scrub_all(Board().generate()).shape[1]
X = np.zeros([data_size, shape], dtype = np.int8)
Y = np.zeros([data_size], dtype = np.float32)

#Test()
Load('data.json')
#Load('buffer.json')
Clear()

model = keras.Sequential([
        keras.layers.Dense(81, activation = 'relu',
                            input_shape = [shape]),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(49, activation = 'relu'),
        keras.layers.Dense(36, activation = 'relu'),
        keras.layers.Dense(25, activation = 'relu'),
        keras.layers.Dense(16, activation = 'relu'),
        keras.layers.Dense(9, activation = 'relu'),
        keras.layers.Dense(4, activation = 'relu'),
        keras.layers.Dense(1)])
model.compile(optimizer = 'adam', loss = 'mse')
model.summary()

Time = time.time()
epoch = 1
#Synthesize()
model.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0)

print("start program")
for epoch in range(epoch, 1_000):
    Save('buffer.json')
    i = 0
    while i < data_size:
        # simulate environment
        state, other = Board(), Board()
        if epoch < 9:
            history = []
            while state.generate().shape[0] and state.reward != 10:
                state, other = other, state
                actions = state.generate()
                action = actions[random.randrange(0, actions.shape[0])]
                history.append(action)

                other.move(action, [0, 0, 1])
                state.move(action, [0, 1, 0])

            for elem in history[:-1 - epoch:-1]:
                other.move(elem, [1, 0, 0])
                state.move(elem, [1, 0, 0])

        actions = state.generate()
        value = model.predict(state.scrub_all(actions), verbose = 0)
        for freedom in range(min(epoch, data_size - i)):
            reward = state.reward
            action = actions[value.argmax() if random.randrange(0, 100) < 95 else random.randrange(0, actions.shape[0])]
            X[i] = state.scrub(action)

            other.move(action, [0, 0, 1])
            state.move(action, [0, 1, 0])

            # replay buffer
            if not state.generate().shape[0] or state.reward == 10:
                Y[i] = reward + discount * state.reward
                i += 1
                break
            else:
                Y[i] = state.reward + discount * np.amax(value)
                state, other = other, state
                actions = state.generate()
                value = model.predict(state.scrub_all(actions), verbose = 0)
                i += 1

            # train model
            if not i % 100:
                Qnew = keras.models.clone_model(model)
                Qnew.compile(optimizer = 'adam', loss = 'mse')
                loss = Qnew.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0).history['loss'][-1]
                model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))

                text = f"loss: {loss}"
                open('log.txt', 'a').write(text + '\n')
                print(text)