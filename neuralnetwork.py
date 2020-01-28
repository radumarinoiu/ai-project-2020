import os
import random
from collections import deque

import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

from keras.optimizers import Adam, Adagrad
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten


class NeuralNetwork(object):
    def __init__(self, name, inputs=None, load=False, learning_rate=0.5, logger=None):
        self.logger = logger
        if load and os.path.exists("model_{}.json".format(name)) and os.path.exists("model_{}.h5".format(name)):
            # load json and create model
            json_file = open("model_{}.json".format(name), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("model_{}.h5".format(name))
            print("Loaded model from disk")
            self.model = loaded_model
        else:
            self.model = Sequential()
            self.model.add(Conv2D(16, (3, 3), input_shape=inputs))
            self.model.add(Conv2D(24, (2, 2)))
            self.model.add(Conv2D(32, (3, 3)))
            self.model.add(Flatten())
            # self.model.add(Dense(80, activation="relu", input_shape=(inputs,)))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dense(32, activation="relu"))
            self.model.add(Dense(1, activation="linear"))
        print(self.model.summary())
        self.model.compile(loss='mse', optimizer=Adagrad(lr=learning_rate), metrics=["mse"])
        self.memory = deque(maxlen=40000)
        self.batch_size = 1024

    def predict(self, inp):
        return self.model.predict(inp)

    def memorize(self, inp, out):
        self.memory.append([inp, out])
        inp = np.rot90(inp, k=1, axes=(0, 1))
        self.memory.append([inp, out])
        inp = np.rot90(inp, k=1, axes=(0, 1))
        self.memory.append([inp, out])
        inp = np.rot90(inp, k=1, axes=(0, 1))
        self.memory.append([inp, out])

    def learn(self):
        # print("Learning...")
        if len(self.memory) > 10000:
            temp_mem = random.sample(self.memory, self.batch_size)
            inputs = np.array([elem[0] for elem in temp_mem])
            labels = np.array([elem[1] for elem in temp_mem])
            history = self.model.fit(
                np.array(inputs),
                np.array(labels),
                epochs=1, verbose=0, batch_size=self.batch_size
            )
            return history.history["mse"][0]
        # print("Learned")
        return 0

    def save(self, name):
        # print("Saving model...")
        model_json = self.model.to_json()
        with open("model_{}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model_{}.h5".format(name))
        print("\nSaved model to disk\n")
