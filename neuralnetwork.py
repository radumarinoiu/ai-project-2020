import os
import random

import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

from collections import deque
from keras.optimizers import Adam, Adagrad
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten


class NeuralNetwork(object):
    def __init__(self, name, inputs=None, load=False, learning_rate=0.5):
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
            self.model.add(Conv2D(32, (3, 3), input_shape=inputs))
            self.model.add(Conv2D(24, (2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(32, activation="relu"))
            self.model.add(Dense(32, activation="relu"))
            self.model.add(Dense(1, activation="linear"))
        print(self.model.summary())
        self.model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
        self.memory = deque(maxlen=20000)
        self.batch_size = 512

    def predict(self, inp):
        return self.model.predict(inp)

    def memorize(self, inp, out, reward, done):
        self.memory.append([inp, out, reward, done])

    def learn(self):
        if len(self.memory) > self.batch_size * 2:
            print("Learning")
            temp_mem = random.sample(self.memory, self.batch_size)

            next_states = np.array([elem[1] for elem in temp_mem])
            next_qs = [elem[0] for elem in self.model.predict(next_states)]

            inputs = []
            labels = []

            for mem_index, (state, _, reward, done) in enumerate(temp_mem):
                if not done:
                    # Partial Q formula
                    new_q = reward + 0.95 * next_qs[mem_index]
                else:
                    new_q = reward

                inputs.append(state)
                labels.append(new_q)

            self.model.fit(
                np.array(inputs),
                np.array(labels),
                epochs=1, verbose=0, batch_size=self.batch_size
            )

    def save(self, name):
        # print("Saving model...")
        model_json = self.model.to_json()
        with open("model_{}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model_{}.h5".format(name))
        print("\nSaved model to disk\n")
