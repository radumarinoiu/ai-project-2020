import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten


class NeuralNetwork(object):
    def __init__(self, name, boardsize=19, load=False, learning_rate=0.1):
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
            self.model.add(Conv2D(16, (5,5), strides=(1, 1), activation='relu', input_shape=(boardsize, boardsize,3)))
            self.model.add(Conv2D(32, (5,5), strides=(1, 1), activation='relu'))
            self.model.add(Conv2D(64, (5,5), strides=(1, 1), activation='relu'))
            self.model.add(Conv2D(1, (1,1), strides=(1, 1), activation='relu'))
            self.model.add(Flatten())
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))
        print(self.model.summary())
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
        self.memory = [[], []]
        
    def prepare_inputs(self, inp):
        array = np.array([inp.board == inp.active_player, inp.board == -inp.active_player, np.full((inp.boardsize, inp.boardsize), (inp.active_player+1)/2)], dtype = np.int32)
        return np.moveaxis(array, 0, -1)

    def predict(self, inp):
        return self.model.predict(np.array([self.prepare_inputs(inp)]))

    def memorize(self, inp, out):
        self.memory[0].append(self.prepare_inputs(inp))
        self.memory[1].append(out)
        if len(self.memory[0]) > 300:
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
