import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json
from keras.layers import Dense


class NeuralNetwork(object):
    def __init__(self, name, inputs=None, load=False, learning_rate=0.5):
        if load:
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
            self.model.add(Dense(100, activation="relu", input_shape=(inputs,)))
            self.model.add(Dense(100, activation="relu"))
            self.model.add(Dense(1, activation="linear"))
        print(self.model.summary())
        self.model.compile(loss='mae', optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
        self.memory = [[], []]

    def predict(self, inp):
        return self.model.predict(np.array([inp.board.flatten()]))

    def memorize(self, inp, out):
        self.memory[0].append(inp.board.flatten())
        self.memory[1].append(out)
        if len(self.memory[0]) > 5000:
            self.learn()

    def learn(self):
        print("Learning...")
        self.model.fit(
            np.array(self.memory[0]),
            np.array(self.memory[1]),
            epochs=1, verbose=0
        )
        self.memory = [[], []]
        print("Learned")

    def save(self, name):
        print("Saving model...")
        model_json = self.model.to_json()
        with open("model_{}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model_{}.h5".format(name))
        print("Saved model to disk\n")
