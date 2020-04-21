import keras
import numpy as np
from keras import Model
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout, Input, Activation, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, History, CSVLogger


class MalariaNet:
    def __init__(self):
        self.learn_rate_smallest = 0.00001
        self.learn_rate_small = 0.0001
        self.learn_rate_medium = 0.001
        self.learn_rate_big = 0.01
        self.dropout = 0.3

        # Epochs
        self.epochs = 25
        # Percentage of training data used for validation
        self.validation_split = 0.2
        self.batch_size = 32
        # Number of epochs without improvement, after which training stops
        self.patience = 5

        # Arguments = (Image size, output size)
        self.model = self.baseline_network((64, 64, 3), 2)
        self.model.summary()
        # model name
        self.filename = 'test'

        self.load_data()

        # Set correct learn rate here
        self.compile_network(learn_rate=self.learn_rate_small)

        self.train_network(self.epochs, self.filename)
        self.evaluate_network()

    def compile_network(self, learn_rate):
        self.model.compile(Adam(lr=learn_rate, decay=learn_rate/10),
                           loss='binary_crossentropy', metrics=['accuracy'])

    def train_network(self, epochs, filename='malariaNet'):
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience),
                     CSVLogger(filename=filename + '_log.csv')]

        training_history = self.model.fit(self.x_train, self.y_train, epochs=epochs,
                                          batch_size=self.batch_size, verbose=1,
                                          validation_split=self.validation_split,
                                          shuffle=True, callbacks=callbacks)
        self.model.save(filename + '_model.h5')
        print('Training finnished of model [' + filename + ']')

        return training_history

    def evaluate_network(self):
        self.score = self.model.evaluate(self.x_test, self.y_test)
        np.save(self.filename + '_score.npy', self.score)
        print(
            'Evaluation of model [' + self.filename + ']: score = ' + str(self.score))

    def load_data(self):
        self.x_train = np.load('./data_set/x_train.npy')
        self.y_train = np.load('./data_set/y_train.npy')

        self.x_test = np.load('./data_set/x_test.npy')
        self.y_test = np.load('./data_set/y_test.npy')

    def preprocess_data(self):
        print('TODO: Implement')

    def baseline_network(self, input_shape, output_shape):
        inputs = Input(input_shape)
        model = Conv2D(32, (3, 3), padding='same')(inputs)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPool2D(2)(model)

        model = Conv2D(32, (3, 3), padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPool2D(2)(model)

        model = Conv2D(64, (3, 3), padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPool2D(2)(model)

        model = Flatten()(model)
        model = Dense(64, activation='relu')(model)
        model = Dropout(self.dropout)(model)

        outputs = Dense(output_shape, activation='softmax')(model)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def smaller_network(self, input_shape, output_shape):
        inputs = Input(input_shape)
        model = Conv2D(16, (3, 3), padding='same')(inputs)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPool2D(2)(model)

        model = Conv2D(32, (3, 3), padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPool2D(2)(model)

        model = Flatten()(model)
        model = Dense(32, activation='relu')(model)
        model = Dropout(self.dropout)(model)

        outputs = Dense(output_shape, activation='softmax')(model)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def bigger_network(self, input_shape, output_shape):
        inputs = Input(input_shape)
        model = Conv2D(32, (3, 3), padding='same')(inputs)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPool2D(2)(model)
        model = Conv2D(128, (3, 3), padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Conv2D(256, (3, 3), padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPool2D(2)(model)

        model = Flatten()(model)
        model = Dropout(self.dropout)(model)
        model = Dense(724, activation='relu')(model)
        model = Dropout(self.dropout)(model)
        model = Dense(724, activation='relu')(model)
        outputs = Dense(output_shape, activation='softmax')(model)

        model = Model(inputs=inputs, outputs=outputs)

        return model


if __name__ == "__main__":
    MalariaNet()
