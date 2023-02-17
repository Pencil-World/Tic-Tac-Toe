from Board import Board
import copy
import json
from tensorflow import keras
import numpy as np
import random
import time

def Load(fstream):
    print("\nLoading Data")

    data = json.load(open('buffer.json', 'r'))
    for i, (key, val) in enumerate(data.items()):
        X[i] = key
        Y[i] = val

    if fstream == 'data.json':
        return

    global epoch, model
    model = keras.models.load_model('model.h5')

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
        state.evaluate()
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
    global state, other, actions
    i = 0
    while i < 5_000:
        reward = state.reward
        action = actions[random.randrange(0, actions.shape[0])]
        backup = copy.deepcopy(X[i])
        X[i] = state.scrub(action)

        other.move(action, [0, 0, 1])
        state.move(action, [0, 1, 0])
        state.evaluate()
    
        if not state.generate().shape[0] or state.reward == 10:
            if state.reward == 10 and not X[i].tolist() in X[:i].tolist():
                i += 1
                X[i] = X[i + 5_000] = X[i - 1]
                Y[i] = Y[i + 5_000] = reward + discount * state.reward

                X[i - 1] = X[i - 1 + 5_000] = backup
                Y[i - 1] = Y[i - 1 + 5_000] = other.reward - discount * state.reward
                i += 1

            state = Board()
            other = Board()
            actions = state.generate()
        else:
            state, other = other, state
            actions = state.generate()
    
    Save('data.json')

Time = time.time()
epoch = 0

#Test()
#Load('data.json')
#Load('buffer.json')
#Clear()

state = Board()
other = Board()
actions = state.generate()
value = np.array([0])

discount = 0.85
data_size = 10_000
shape = state.scrub_all(actions).shape[1]
X = np.zeros([data_size, shape], dtype = np.int8)
Y = np.zeros([data_size], dtype = np.float32)

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
model.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0)

Synthesize()
print("start program")
for epoch in range(epoch, 1_000):
    Save('buffer.json')
    history = []
    for i in range(data_size):
        # simulate environment
        reward = state.reward
        action = actions[value.argmax() if random.randrange(0, 100) < min(85, epoch * 10) else random.randrange(0, actions.shape[0])]
        X[i] = state.scrub(action)

        other.move(action, [0, 0, 1])
        state.move(action, [0, 1, 0])
        state.evaluate()

        # replay buffer
        if not state.generate().shape[0] or state.reward == 10:
            Y[i] = reward + discount * state.reward
            Y[i - 1] = other.reward - discount * state.reward

            state = Board()
            other = Board()

            actions = state.generate()
            value = model.predict(state.scrub_all(actions), verbose = 0)
        else:
            state, other = other, state
            actions = state.generate()
            value = model.predict(state.scrub_all(actions), verbose = 0)
            if actions.shape[0] < 8:
                Y[i - 1] = state.reward + discount * np.amax(value)

        # train model. is it training enough?
        if not (i + 1) % 100:
            Qnew = keras.models.clone_model(model)
            Qnew.compile(optimizer = 'adam', loss = 'mse')
            loss = Qnew.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0).history['loss'][-1]
            model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))

            text = f"loss: {loss}"
            open('log.txt', 'a').write(text + '\n')
            print(text)