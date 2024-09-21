import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LinearClassifier:
    def __init__(self, fpath: str, labelcolumn: int = 0):
        # Data - matrix with shape (n, d+1) since one column is labels
        self.data = np.genfromtxt(fpath, delimiter=",")
        self.n = self.data.shape[0]

        # Labels (n, 1) and feature vectors (n, d)
        self.labels = np.array(self.data[:, labelcolumn], int)
        self.features = np.hstack((self.data[:, :labelcolumn], self.data[:, labelcolumn+1:]))
        print(f"Labels shape:\t{self.labels.shape}")

        # Add bias - feature vectors are now all (1, d+1), so feature matrix is (n, d+1)
        self.features = np.hstack((self.features, np.ones((self.n, 1))))
        print(f"Features shape:\t{self.features.shape}")

        # Weights - vertical vector with shape (d+1, 1)
        self.weights = np.zeros(self.features.shape[1])
        print(f"Weights shape:\t{self.weights.shape}")

        # Hyperparamters
        self.epochs = int(10e3)
        self.lr = lambda k: 1/(1 + k)
        self.epsilon = 10e-6

        # Monitoring
        self.losses = []

    def fit(self):
        print("Training----")
        for ep in range(self.epochs):
            cum_loss = 0

            # Shuffle
            shuffler = np.random.permutation(self.n)
            tlabels = self.labels[shuffler]
            tfeatures = self.features[shuffler, :]

            for k in range(self.n):
                # print(f"Feature {k}:\t{tfeatures[k, :]}")
                loss = tlabels[k] * (self.weights.T @ tfeatures[k, :])
                # print(f"Loss {k}:\t{loss}")
                # print(f"Loss shape:\t{loss.shape}")
                if loss <= 1:
                    cum_loss += loss
                    delta = self.lr(k) * tlabels[k] * tfeatures[k, :]
                    # print(delta.T)
                    # print(self.weights)
                    self.weights += delta.T
                    # print(f"Updated weights:\t{self.weights}")
                # print("--")
            self.losses.append(cum_loss)
            if ep % 5 == 0:
                print(cum_loss)
            if len(self.losses) > 1 and abs(self.losses[-1] - self.losses[-2]) < self.epsilon:
                break

    def plot(self):
        sns.scatterplot(x=self.features[:, 0], y=self.features[:, 1], hue=self.labels, palette="Set2")

        tmpx = np.linspace(0, 10)
        tmpy = -self.weights.T[1]/self.weights.T[0] * tmpx + self.weights.T[2]

        sns.lineplot(x=tmpx, y=tmpy)
        plt.show()


if __name__ == '__main__':
    lc = LinearClassifier(f"{__file__}/../../Data/linearly_sep_binary_classification.csv")
    lc.fit()
    lc.plot()
