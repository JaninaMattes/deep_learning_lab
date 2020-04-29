import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

class VisualizeData:
    def __init__(self):
        self.x_train = np.load('./data_set/x_train.npy')
        self.y_train = np.load('./data_set/y_train.npy')

        self.x_test = np.load('./data_set/x_test.npy')
        self.y_test = np.load('./data_set/y_test.npy')

        offset = 700
        for i in range(10):
            plt.imshow(self.x_train[offset + i, ])
            plt.show()
            print('Label ' + str(offset+i) + ' = ' +
                  str(self.y_train[offset+i]))

        for i in range(10):
            plt.imshow(self.x_test[offset + i, ])
            plt.show()
            print('Label ' + str(offset+i) +
                  ' = ' + str(self.y_test[offset+i]))


if __name__ == "__main__":
    VisualizeData()
