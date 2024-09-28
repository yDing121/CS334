"""
Perceptron Algorithm with Offset
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""
import pip
pip.main(["install", "numpy", "pandas", "scikit-learn"])

import numpy as np
from helper import load_data

def all_correct(X, y, theta, b):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,), normal vector of decision boundary
        b: float, offset

    Returns true if the linear classifier specified by theta and b correctly classifies all examples
    """

    # If label and prediction match, then the product will be positive. Check if all label*predictions are positive
    res = y * (X @ theta + np.full((y.size, ), b))
    return np.all(res > 0)


def perceptron(X, y):
    """
    Implements the Perception algorithm for binary linear classification.
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)

    Returns:
        theta: np.array, shape (d,)
        b: float
        alpha: np.array, shape (N,)
            Misclassification vector, in which the i-th element is has the number of times 
            the i-th point has been misclassified)
    """

    N = X.shape[0]
    d = X.shape[1]

    assert N == y.size and N == y.shape[0]

    theta = np.zeros((d, ))
    b = 0.0
    alpha = np.zeros((N, ))

    # Add a hard stop in case something is weird with all_correct
    maxiters = 1e4
    while not all_correct(X, y, theta, b) and maxiters > 0:
        maxiters -= 1
        for i in range(N):
            if y[i] * (theta @ X[i, :] + b) <= 0:
                theta += y[i] * X[i, :]
                alpha[i] += 1
                b += y[i]


    return theta, b, alpha


def main(fname):
    X, y = load_data(fname, d=2)
    theta, b, alpha = perceptron(X, y)

    # # Uncomment to see question (e) in action - we get a faster convergence.
    # from sklearn.utils import shuffle
    # X_s, y_w = shuffle(X, y, random_state=0)
    # theta, b, alpha = perceptron(X, y)

    print("Done!")
    print("============== Classifier ==============")
    print("Theta: ", theta)
    print("b: ", b)

    print("\n")
    print("============== Alpha ===================")
    print("i \t Number of Misclassifications")
    print("========================================")
    for i in range(len(alpha)):
        print(i, "\t\t", alpha[i])
    print("Total Number of Misclassifications: ", np.sum(alpha))


if __name__ == '__main__':
    main("dataset/classification.csv")
    X, y = load_data("dataset/classification.csv", d=2)
    theta, b, alpha = perceptron(X, y)

    print("="*20)
    print(f"Theta with alpha sum:\t{y * alpha @ X}")
    print(f"Theta from perceptron:\t{theta}")
    print("="*20)
    print(f"b with alpha sum:\t{y @ alpha}")
    print(f"b from perceptron:\t{b}")