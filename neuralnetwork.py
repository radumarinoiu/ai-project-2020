import numpy as np
from keras.models import model_from_json, Model
from keras.layers import Dense, Conv2D, Flatten, Input, Concatenate
from keras.optimizers import RMSprop


class NeuralNetwork(object):
    def __init__(self, name, boardsize=19, load=False, learning_rate=0.00002):
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
            inputBoard = Input(shape=(boardsize, boardsize, 4,))
            inputScore = Input(shape=(1,))
            conv1 = Conv2D(16, (3,3), strides=(1, 1), activation='relu')(inputBoard)
            conv2 = Conv2D(32, (3,3), strides=(1, 1), activation='relu')(conv1)
            conv3 = Conv2D(64, (3,3), strides=(1, 1), activation='relu')(conv2)
            flat = Flatten()(conv3)
            combined = Concatenate()([flat, inputScore])
            dense = Dense(32, activation='relu')(combined)
            out = Dense(1, activation='sigmoid')(dense)
            self.model = Model(inputs=[inputBoard, inputScore], outputs=out)
        print(self.model.summary())
        self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=learning_rate), metrics=["accuracy"])
        self.memory = [[], [], []]
        
    def prepare_inputs_black(self, inp):
        array = np.array([inp.board == 1, 
                          inp.board == -1, 
                          np.full((inp.boardsize, inp.boardsize), (inp.active_player+1)/2), 
                          np.full((inp.boardsize, inp.boardsize), int(inp.passed))], dtype = np.int32)
        return np.moveaxis(array, 0, -1)
    
    def prepare_inputs_white(self, inp):
        array = np.array([inp.board == -1, 
                          inp.board == 1, 
                          np.full((inp.boardsize, inp.boardsize), (-inp.active_player+1)/2), 
                          np.full((inp.boardsize, inp.boardsize), int(inp.passed))], dtype = np.int32)
        return np.moveaxis(array, 0, -1)

    def predict(self, inp):
        return self.model.predict([np.array([self.prepare_inputs_black(inp)]), np.array([inp.score()])])
    
    def memorize_rotated(self, inp, score, out):
        self.memory[0].append(inp)
        self.memory[1].append(score)
        self.memory[2].append(out)
        inp = np.rot90(inp, k=1, axes=(0, 1))
        self.memory[0].append(inp)
        self.memory[1].append(score)
        self.memory[2].append(out)
        inp = np.rot90(inp, k=1, axes=(0, 1))
        self.memory[0].append(inp)
        self.memory[1].append(score)
        self.memory[2].append(out)
        inp = np.rot90(inp, k=1, axes=(0, 1))
        self.memory[0].append(inp)
        self.memory[1].append(score)
        self.memory[2].append(out)
        
    def memorize_flipped_rotated(self, inp, score, out):
        self.memorize_rotated(inp, score, out)
        self.memorize_rotated(np.flip(inp, axis = 0), score, out)

    def memorize(self, inp, out): #go is a game symmetrical to rotation, flipping of the board and exchange of colors
        self.memorize_flipped_rotated(self.prepare_inputs_black(inp), inp.score(), out)
        self.memorize_flipped_rotated(self.prepare_inputs_white(inp), -inp.score(), 1-out)
        if len(self.memory[0]) > 10000:
            self.learn()

    def learn(self):
        print("Learning...")
        self.model.fit(
            [np.array(self.memory[0]),np.array(self.memory[1])],
            np.array(self.memory[2]),
            epochs=1, verbose=0
        )
        self.memory = [[], [], []]
        print("Learned")

    def save(self, name):
        print("Saving model...")
        model_json = self.model.to_json()
        with open("model_{}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model_{}.h5".format(name))
        print("Saved model to disk\n")
