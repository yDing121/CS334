"""
Perceptron Algorithm with Offset
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

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
    # TODO: Implement this function
    return True


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
    # TODO: Implement this function
    theta = ???
    b = ???
    alpha = ???
    return theta, b, alpha


def main(fname):
    X, y = load_data(fname, d=2)
    theta, b, alpha = perceptron(X, y)

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
