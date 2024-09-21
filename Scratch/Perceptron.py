import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, data: np.ndarray, lr=0.05):
        self.n = data.shape[0]
        self.features = np.hstack((data[:, 1:], np.ones((self.n, 1))))
        self.theta = np.zeros((self.features.shape[1], 1))
        self.labels = data[:, :1]
        self.lr = lr

        print(f"Features dim:\t{self.features.shape}")
        print(f"Theta dim:\t{self.theta.shape}")
        print(f"Labels dim:\t{self.labels.shape}")
        print(f"Learning rate:\t{self.lr}")

    def fit(self):
        cont = True

        while cont:
            preds = self.features.dot(self.theta.T)
            print(preds)
            cont = False





if __name__ == '__main__':
    data = np.genfromtxt(f"{__file__}/../../Data/linearly_sep_binary_classification.csv", delimiter=",")
    # print(data)
    p = Perceptron(data)
    p.fit()
