#from Board import Board
#import datetime
#import json
#from tensorflow import keras
#import numpy as np
#import random
#import time

#def Clear():
#    open('log.txt', 'w').close()
#    open('buffer.json', 'w').close()

#def Load(fstream):
#    with open('debugger.txt', 'a') as debugger:
#        debugger.write("Loading Data\n")
#    global epoch, i, lim, model

#    data = json.load(open(fstream, 'r'))
#    length = len(data.items())
#    X[:length] = [np.array(json.loads(elem)) for elem in data.keys()]
#    Y[:length] = list(data.values())

#    for it, _i in enumerate(range(length, data_size)):
#        X[_i] = X[it]
#        Y[_i] = Y[it]

#    if fstream == 'data.json':
#        model.compile(optimizer = 'adam', loss = 'mse')
#        model.fit(X, Y, batch_size = None, epochs = 100, verbose = 0, shuffle = True)
#        return

#    f = open('log.txt', 'r')
#    log = f.read().split()
#    f.close()

#    model = keras.models.load_model('model.h5')
#    index = -log[::-1].index("epoch:")
#    epoch = int(log[index])
#    i = lim = (epoch - 1) * cluster_size % data_size

#    f = open('log.txt', 'w')
#    f.write(''.join([elem + ('\n' if elem.find(':') == -1 else ' ') for elem in log[:index - 1]]))
#    f.close()

#def Save(fstream):
#    with open('debugger.txt', 'a') as debugger:
#        debugger.write("Saving Data\n")
   
#        JSON = dict(zip([repr(elem.tolist()) for elem in X], Y))
#        json.dump(JSON, open(fstream, 'w'), indent = 4)

#        if fstream == 'data.json':
#            return

#        model.save('model.h5')
#        text = f"epoch: {epoch}\ntime: {time.time() - Time}\n"
#        open('log.txt', 'a').write(text)
#        debugger.write(text)

#def Synthesize():
#    print("Synthesizing Data")
#    state = Board()
#    i = 0
#    while True:
#        history = []
#        isModel = False
#        actions = state.generate()

#        for temp in range(2 * (data_size - i)):
#            action = actions[random.randrange(0, actions.shape[0])]
#            history.append(action)

#            state.move(action, [0, 0, 1] if isModel else [0, 1, 0])
#            actions = state.generate()

#            if not actions.shape[0] or state.reward == 10:
#                break
#            isModel = not isModel

#        reward = 10 if isModel else 1
#        for action in history[::-1]:
#            state.move(action, [1, 0, 0])

#            if isModel:
#                reward = state.reward + discount * reward
#                X[i] = state.scrub(action)
#                Y[i] = reward

#            # train model
#            i += isModel
#            if isModel and not i % 100:
#                print(Y[i - 100:i])
#                if i == data_size:
#                    Save('data.json')
#                    return
#            isModel = not isModel

#def Test():
#    print("\nTest")
#    model = keras.models.load_model('model.h5')

#    table = np.zeros([9, 9])
#    for i, elem in enumerate(table):
#        elem[i] = 1

#    state = Board()
#    PlayersTurn = True
#    for temp in range(1_000):
#        print()
#        print(state)

#        actions = state.generate()
#        if PlayersTurn:
#            action = table[int(input())]
#            state.move(action, [0, 1, 0])
#        else:
#            value = model.predict(state.scrub_all(actions), verbose = 0)
#            action = actions[value.argmax() if random.randrange(0, 100) < 95 else random.randrange(0, actions.shape[0])]
#            state.move(action, [0, 0, 1])

#        PlayersTurn = not PlayersTurn
#        if not state.generate().shape[0] or state.reward == 10:
#            print()
#            print(state)
#            state = Board()
#            PlayersTurn = True

#open('debugger.txt', 'w').close()
#Time = time.time()
#epoch = 1
#i = lim = 0
#alpha = 1 / 100 # take the average across the past 100 samples

#discount = 0.85
#data_size = 10_000
#cluster_size = 1_000
#shape = Board().scrub_all(Board().generate()).shape[1]
#X = np.zeros([data_size, shape], dtype = np.int8)
#Y = np.zeros([data_size], dtype = np.float64)

#model = keras.Sequential([
#        keras.layers.Dense(125, activation = 'relu',
#                            input_shape = [shape]),
#        keras.layers.Dense(125, activation = 'relu'),
#        keras.layers.Dense(125, activation = 'relu'),
#        keras.layers.Dense(25, activation = 'relu'),
#        keras.layers.Dense(5, activation = 'relu'),
#        keras.layers.Dense(1)])
#model.summary()

#Synthesize()
#Test()
#Load('data.json')
#Clear()
##Load('buffer.json')

#state = Board()
#with open('debugger.txt', 'a') as debugger:
#    debugger.write("start program\n")
#for epoch in range(epoch, 1_000):
#    Save('buffer.json')

#    if i == data_size:
#        i = lim = 0
#    lim += cluster_size
#    WinLossRatio = [0, 0]
#    while i < lim:
#        # simulate environment
#        history = []
#        isModel = False
#        actions = state.generate()

#        for temp in range(2 * (data_size - i)):
#            action = actions[value.argmax() if isModel and random.randrange(0, 100) < min(95, epoch * 100 // 25) else random.randrange(0, actions.shape[0])]
#            history.append(action)

#            state.move(action, [0, 0, 1] if isModel else [0, 1, 0])
#            actions = state.generate()

#            if not actions.shape[0] or state.reward == 10:
#                if state.reward == 10:
#                    WinLossRatio[not isModel] += 1
#                break
#            else:
#                isModel = not isModel
#                if isModel:
#                    value = model.predict(state.scrub_all(actions), verbose = 0)

#        # replay buffer
#        reward = 10 if isModel else 1
#        for action in history[::-1]:
#            state.move(action, [1, 0, 0])

#            if isModel:
#                reward = state.reward + discount * reward
#                X[i] = state.scrub(action)
#                mean = model.predict(np.array([X[i]]), verbose = 0)
#                Y[i] = mean + alpha * (reward - mean)

#            # train model
#            i += isModel
#            if isModel and not i % 100:
#                Qnew = keras.models.clone_model(model)
#                Qnew.compile(optimizer = 'adam', loss = 'mse')
#                loss = Qnew.fit(X, Y, batch_size = None, epochs = 100, verbose = 0, shuffle = True).history['loss'][-1]
#                model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))

#                text = f"loss: {loss}\n"
#                open('log.txt', 'a').write(text)
#                with open('debugger.txt', 'a') as debugger:
#                    debugger.write(f"{datetime.datetime.now()} " + text)

#            isModel = not isModel

#    with open('debugger.txt', 'a') as debugger:
#        debugger.write(f"win to loss ratio (expected to be between 50% and 90%): {WinLossRatio[0] * 100 / sum(WinLossRatio)} percent\n")

#from Board import Board
#import datetime
#import json
#from tensorflow import keras
#import numpy as np
#import random
#import time

#def Load(fstream):
#    with open('debugger.txt', 'a') as debugger:
#        debugger.write("Loading Data\n")
#    global epoch, i, lim, model

#    data = json.load(open(fstream, 'r'))
#    length = len(data.items())
#    X[:length] = [np.array(json.loads(elem)) for elem in data.keys()]
#    Y[:length] = list(data.values())

#    for it, _i in enumerate(range(length, data_size)):
#        X[_i] = X[it]
#        Y[_i] = Y[it]

#    if fstream == 'data.json':
#        model.compile(optimizer = 'adam', loss = 'mse')
#        model.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0)
#        return

#    f = open('log.txt', 'r')
#    log = f.read().split()
#    f.close()

#    model = keras.models.load_model('model.h5')
#    index = -log[::-1].index("epoch:")
#    epoch = int(log[index])
#    i = lim = (epoch - 1) * cluster_size % data_size

#    f = open('log.txt', 'w')
#    f.write(''.join([elem + ('\n' if elem.find(':') == -1 else ' ') for elem in log[:index - 1]]))
#    f.close()

#    #for _i in range(10):
#    #    sentries[_i] = keras.models.load_model(str(_i) + '.h5')

#def Save(fstream):
#    with open('debugger.txt', 'a') as debugger:
#        debugger.write("Saving Data\n")
   
#        JSON = dict(zip([repr(elem.tolist()) for elem in X], Y))
#        json.dump(JSON, open(fstream, 'w'), indent = 4)

#        if fstream == 'data.json':
#            return

#        model.save('model.h5')
#        text = f"epoch: {epoch}\ntime: {time.time() - Time}\n"
#        open('log.txt', 'a').write(text)
#        debugger.write(text)

#    #for _i in range(10):
#    #    sentries[_i].save(str(_i) + '.h5')

#def Test():
#    print("\nTest")
#    model = keras.models.load_model('model.h5')

#    state, other = Board(), Board()
#    table = np.zeros([9, 9])
#    for i, elem in enumerate(table):
#        elem[i] = 1

#    for i in range(1_000):
#        print()
#        print(state if i % 2 else other)

#        actions = state.generate()
#        value = model.predict(state.scrub_all(actions), verbose = 0)
#        action = actions[value.argmax() if random.randrange(0, 100) < 95 else random.randrange(0, actions.shape[0])]
#        if i % 2:
#            action = table[int(input())]

#        other.move(action, [0, 0, 1])
#        state.move(action, [0, 1, 0])

#        if not state.generate().shape[0] or state.reward == 10:
#            print()
#            print(state if i % 2 else other)
#            state, other = Board(), Board()
#        state, other = other, state

#def Clear():
#    open('log.txt', 'w').close()
#    open('buffer.json', 'w').close()

#def Synthesize():
#    print("Synthesizing Data")
#    i = 0
#    while True:
#        history = []
#        state, other = Board(), Board()
#        actions = state.generate()
#        while actions.shape[0] and state.reward != 10:
#            state, other = other, state
#            action = actions[random.randrange(0, actions.shape[0])]
#            history.append(action)

#            other.move(action, [0, 0, 1])
#            state.move(action, [0, 1, 0])
#            actions = state.generate()

#        for it, action in enumerate(history[::-1]):
#            reward = state.reward
#            other.move(action, [1, 0, 0])
#            state.move(action, [1, 0, 0])

#            X[i] = state.scrub(action)
#            Y[i] = state.reward + discount * (reward if it < 2 else Y[i - 2])

#            state, other = other, state
#            i += 1
#            if not i % 100:
#                print(Y[i - 100:i])
#                if i == data_size:
#                    Save('data.json')
#                    return

#open('debugger.txt', 'w').close()
#Time = time.time()
#epoch = 1
#i = lim = 0
#sentries = [None] * 10

#discount = 0.85
#data_size = 50_000
#cluster_size = 1_000
#shape = Board().scrub_all(Board().generate()).shape[1]
#X = np.zeros([data_size, shape], dtype = np.int8)
#Y = np.zeros([data_size], dtype = np.float64)

#model = keras.Sequential([
#        keras.layers.Dense(125, activation = 'relu',
#                            input_shape = [shape]),
#        keras.layers.Dense(125, activation = 'relu'),
#        keras.layers.Dense(125, activation = 'relu'),
#        keras.layers.Dense(125, activation = 'relu'),
#        keras.layers.Dense(125, activation = 'relu'),
#        keras.layers.Dense(125, activation = 'relu'),
#        keras.layers.Dense(125, activation = 'relu'),
#        keras.layers.Dense(25, activation = 'relu'),
#        keras.layers.Dense(5, activation = 'relu'),
#        keras.layers.Dense(1)])
#model.summary()
#for _i in range(10):
#    sentries[_i] = keras.models.clone_model(model)

##Test()
#Load('data.json')
#Clear()
##Load('buffer.json')
##Synthesize()
## multi agent reinforcement learning or stochiactic environment. differentiate between playing first and playing second. 2 different sets of models. like how black going first in go is a benefit. and white gets a handicap. 
#with open('debugger.txt', 'a') as debugger:
#    debugger.write("start program\n")
#for epoch in range(epoch, 1_000):
#    Save('buffer.json')

#    #sentries = sentries[1:] + sentries[:1]
#    #sentries[0] = keras.models.clone_model(model)
#    #sentries[0].set_weights(model.get_weights())

#    if i == data_size:
#        i = lim = 0
#    lim += cluster_size
#    WinLossRatio = [0, 0]
#    while i < lim:
#        # simulate environment
#        state, other = Board(), Board()
#        #paragon = sentries[random.randrange(0, min(epoch, 10))]
#        isModel = True
#        if random.randint(0, 1):
#            actions = state.generate()
#            action = actions[random.randrange(0, actions.shape[0])]
#            other.move(action, [0, 1, 0])
#            state.move(action, [0, 0, 1])

#        # replay buffer
#        actions = state.generate()
#        value = model.predict(state.scrub_all(actions), verbose = 0)
#        for temp in range(data_size - i + 1):
#            action = actions[value.argmax() if random.randrange(0, 100) < min(95, epoch * 100 // 25) and isModel else random.randrange(0, actions.shape[0])]
#            #action = actions[value.argmax() if random.randrange(0, 100) < 90 else random.randrange(0, actions.shape[0])]
#            if isModel:
#                reward = state.reward
#                X[i] = state.scrub(action)

#            other.move(action, [0, 0, 1])
#            state.move(action, [0, 1, 0])
#            actions = state.generate()

#            flag = False
#            if not actions.shape[0] or state.reward == 10:
#                if state.reward == 10:
#                    WinLossRatio[not isModel] += 1
#                Y[i] = reward + discount * (state.reward if isModel else other.reward)
#                flag = True
#            else:
#                state, other = other, state
#                isModel = not isModel
#                #value = (model if isModel else paragon).predict(state.scrub_all(actions), verbose = 0)
#                if isModel:
#                    value = model.predict(state.scrub_all(actions), verbose = 0)
#                    Y[i] = reward + discount * value.max()
#                    flag = True

#            # train model
#            i += flag
#            if flag and not i % 100:
#                Qnew = keras.models.clone_model(model)
#                Qnew.compile(optimizer = 'adam', loss = 'mse')
#                loss = Qnew.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0).history['loss'][-1]
#                model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))

#                text = f"loss: {loss} time: {time.time() - Time}\n"
#                open('log.txt', 'a').write(text)
#                with open('debugger.txt', 'a') as debugger:
#                    debugger.write(f"{datetime.datetime.now()} " + text)

#            if not actions.shape[0] or state.reward == 10:
#                break

#    with open('debugger.txt', 'a') as debugger:
#        debugger.write(f"win to loss ratio (expected to be between 50% and 90%): {WinLossRatio[0] * 100 / sum(WinLossRatio)} percent\n")