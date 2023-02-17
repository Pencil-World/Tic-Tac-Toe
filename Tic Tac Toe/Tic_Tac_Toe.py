# from Board import Board
# import copy
# import json
# from tensorflow import keras
# import numpy as np
# import random
# import time

# does this even work? 
def Load(fstream):
    print("\nLoading Data")

    data = json.load(open('buffer.json', 'r'))
    for i, (key, val) in enumerate(data.items()):
        X1[i] = key
        Y1[i] = val

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
        JSON[np.array2string(X1[i], separator = ", ", max_line_width = 1_000)] = float(Y1[i])
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
    global state, other
    i = 0
    while i < 5_000:
        reward = state.reward
        action = state.generate()[random.randrange(0, actions.shape[0])]
        X2[i] = X1[i]
        X1[i] = state.scrub(action)

        other.move(action, [0, 0, 1])
        state.move(action, [0, 1, 0])
        state.evaluate()
    
        if not state.generate().shape[0] or state.reward == 10:
            if state.reward == 10:
                #if not X1[i].tolist() in X1[:i].tolist():
                Y1[i] = Y1[i + 5_000] = reward + discount * state.reward
                i += 1
                #if not backup.tolist() in X1[:i].tolist():
                Y2[i] = Y2[i + 5_000] = other.reward - discount * state.reward# remove!!!
                # the answer to life is to set the default reward from evaluate to 1
                # this disencourages the machine to lose and it also encourages the machine to continue the game
                # with discount at 0.85, it will always be advantagous to win the game over extending it
                i += 1

            state = Board()
            other = Board()
        else:
            state, other = other, state
    
    #Save('data.json')

# print("Initial state: reward = 0 \n", Board([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
# print("Scrub O move at [2][0]\n", Board([[0, 0, 0], [0, 0, 0], [1, 0, 0]]))
# print("Scrub X move at [2][2]\n", Board([[0, 0, 0], [0, 0, 0], [1, 0, 2]]))
# print("-8.5 is O move at [0][2]\n", Board([[0, 0, 0], [1, 0, 0], [1, 0, 2]]))
# print("8.5 is X move at [0][0]\n", Board([[0, 0, 0], [1, 0, 2], [1, 0, 2]]))
# print("Terminal state: reward = 10\n", Board([[1, 0, 0], [1, 0, 2], [1, 0, 2]]))

#Test()
#Load('data.json')
#Load('buffer.json')
#Clear()

discount = 0.85
data_size = 10_000
shape = Board().scrub_all(Board().generate()).shape[1]
X1 = np.zeros([data_size, shape], dtype = np.int8)
Y1 = np.zeros([data_size], dtype = np.float32)
X2 = np.zeros([data_size, shape], dtype = np.int8)
Y2 = np.zeros([data_size], dtype = np.float32)

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
epoch = 0
Synthesize()
model.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0)
win.fit(X1, Y1, batch_size = 64, epochs = 100, verbose = 0)
lose.fit(X2, Y2, batch_size = 64, epochs = 100, verbose = 0)

print("start program")

for epoch in range(1_000):
    Save('buffer.json')
    history = []
    #freedom = 1 # switch epoch to freedom
    i = 0
    while i < data_size:
        # simulate environment
        state, other = Board(), Board()
        actions = state.generate()
        if epoch < 10:
            while actions.shape[0] and state.reward != 10:
                state, other = other, state
                action = actions[random.randrange(0, actions.shape[0])]
                history.append(action)

                other.move(action, [0, 0, 1])
                state.move(action, [0, 1, 0])
                state.evaluate()
                actions = other.generate()
            for elem in history[:epoch:-1]:
                other.move(elem, [1, 0, 0])
                state.move(elem, [1, 0, 0])

        for freedom in range(min(epoch, data_size - i)):
            reward = state.reward
            action = actions[value.argmax() if random.randrange(0, 100) < 75 else random.randrange(0, actions.shape[0])]
            history.append(action)
            X1[i] = state.scrub(action)
            X2[i] = state.scrub(action)

            other.move(action, [0, 0, 1])
            state.move(action, [0, 1, 0])
            state.evaluate()

            # replay buffer
            if not state.generate().shape[0] or state.reward == 10:
                Y1[i] = reward + discount * state.reward
                Y2[i] = other.reward - discount * discount * state.reward
                break
            else:
                Y1[i] = state.reward + discount * np.amax(winVal)
                state, other = other, state
                actions = state.generate()
                winVal = win.predict(state.scrub_all(actions), verbose = 0)
                loseVal = lose.predict(state.scrub_all(actions), verbose = 0)
                value = winVal + loseVal
                if actions.shape[0] < 8:
                    Y2[i] = state.reward + discount * np.amax(loseVal)

            # train model. is it training enough?
            i += 1
            if not (i + 1) % 100:
                Qwin = keras.models.clone_model(model)
                Qwin.compile(optimizer = 'adam', loss = 'mse')
                loss = Qwin.fit(X1, Y1, batch_size = 64, epochs = 100, verbose = 0).history['loss'][-1]

                Qlose = keras.models.clone_model(model)
                Qlose.compile(optimizer = 'adam', loss = 'mse')
                loss = Qlose.fit(X2, Y2, batch_size = 64, epochs = 100, verbose = 0).history['loss'][-1]

                win.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qwin.get_weights(), dtype = object))
                lose.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qlose.get_weights(), dtype = object))

                text = f"loss: {loss}"
                open('log.txt', 'a').write(text + '\n')
                print(text)

for epoch in range(epoch, 1_000):
    Save('buffer.json')
    for i in range(data_size):
        # simulate environment
        reward = state.reward
        action = actions[value.argmax() if random.randrange(0, 100) < 95 else random.randrange(0, actions.shape[0])]
        X1[i] = state.scrub(action)

        other.move(action, [0, 0, 1])
        state.move(action, [0, 1, 0])
        state.evaluate()

        # replay buffer
        if not state.generate().shape[0] or state.reward == 10:
            Y1[i] = reward + discount * state.reward
            Y1[i - 1] = other.reward - discount * discount * state.reward

            state = Board()
            other = Board()

            actions = state.generate()
            value = model.predict(state.scrub_all(actions), verbose = 0)
        else:
            state, other = other, state
            actions = state.generate()
            value = model.predict(state.scrub_all(actions), verbose = 0)
            if actions.shape[0] < 8:
                Y1[i - 1] = state.reward + discount * np.amax(value)

        # train model. is it training enough?
        if not (i + 1) % 100:
            Qnew = keras.models.clone_model(model)
            Qnew.compile(optimizer = 'adam', loss = 'mse')
            loss = Qnew.fit(X1, Y1, batch_size = 64, epochs = 100, verbose = 0).history['loss'][-1]
            model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))

            text = f"loss: {loss}"
            open('log.txt', 'a').write(text + '\n')
            print(text)