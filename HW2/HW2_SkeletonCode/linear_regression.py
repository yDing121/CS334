"""
Linear Regression
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from fontTools.ttLib.tables.E_B_D_T_ import ebdt_bitmap_format_6
import seaborn as sns
from numpy.f2py.crackfortran import true_intent_list

np.random.seed(69)

from helper import load_data
import time

def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector
    i.e. [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (N, 1). Each row is one instance.
        M: a non-negative integer

    Returns:
        Phi: np.array, shape (N, M+1)
    """
    Phi = np.vander(X.T[0], N=M+1, increasing=True)

    return Phi

def calculate_squared_loss(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        theta: np.array, shape (d,)

    Returns:
        loss: float. The empirical risk based on squared loss as defined in the assignment.
    """
    assert X.shape[0] == y.size and X.shape[1] == theta.size
    n = X.shape[0]
    loss = 1/(2 * n) * ((y - X @ theta) @ (y - X @ theta))
    return loss

def calculate_RMS_Error(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        theta: np.array, shape (d,)

    Returns:
        E_rms: float. The root mean square error as defined in the assignment.
    """
    assert X.shape[0] == y.size and X.shape[1] == theta.size
    n = X.shape[0]

    E_rms = np.sqrt((1/n) * ((y - X @ theta) @ (y - X @ theta)))
    return E_rms


def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        learning_rate: float, the learning rate for GD

    Returns:
        theta: np.array, shape (d,)
    """
    assert X.shape[0] == y.size
    n = X.shape[0]
    d = X.shape[1]

    theta = np.zeros((d,))
    MAX_ITERS = 1e6
    k = 0
    prev_loss = 0
    cur_loss = 1

    # # Logging
    # losses = []

    while k < MAX_ITERS and abs(cur_loss - prev_loss) > 1e-10:
        # Set up next cycle
        prev_loss = cur_loss
        cur_loss = calculate_squared_loss(X, y, theta)

        # # Log losses - optional
        # losses.append(cur_loss)

        for i in range(n):
            # Update theta
            theta = theta + learning_rate * (y[i] - theta @ X[i, :]) * X[i, :]

        k += 1

    # # Track losses - optional
    # sns.lineplot(x=[i for i in range(k)], y=losses)
    # plt.show()
    print(f"Iterations:\t{k}")
    return theta


def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10

    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        learning_rate: float or 'adaptive', the learning rate for SGD

    Returns:
        theta: np.array, shape (d,)
    """
    assert X.shape[0] == y.size
    n = X.shape[0]
    d = X.shape[1]

    if str(learning_rate).lower() == 'adaptive':
        adaptive = True
    else:
        adaptive = False

    k = 0
    theta = np.zeros((d, ))
    cur_loss = 0
    prev_loss = 1
    MAX_ITERS = 1e6

    # Logging
    # losses = []

    while k < MAX_ITERS and abs(cur_loss - prev_loss) > 1e-10:
        prev_loss = cur_loss
        cur_loss = calculate_squared_loss(X, y, theta)

        # Dynamic lr
        if adaptive:
            learning_rate = 1/(1+k)

        # # Logging
        # losses.append(cur_loss)

        # Use a permutation of length n to determine the order we process the data
        perm = np.random.permutation(n)
        for i in perm:
            theta = theta + learning_rate * (y[i] - theta @ X[i, :]) * X[i, :]

        k += 1

    # # Logging
    # sns.lineplot(x=[i for i in range(k)], y=losses)
    # plt.show()

    print(f"Iterations:\t{k}")

    return theta


def ls_closed_form_solution(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    assert X.shape[0] == y.size
    d = X.shape[1]

    # theta = np.linalg.inv(X.T @ X) @ X.T @ y
    # theta = np.linalg.pinv(X) @ y
    theta = np.linalg.inv(X.T @ X + reg_param * np.eye(d, d)) @ X.T @ y
    return theta


def weighted_ls_closed_form_solution(X, y, weights, reg_param=0):
    """
    Implements the closed form solution for weighted least squares regression.

    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        weights: np.array, shape (N,), the weights for each data point
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # # TODO: Implement this function
    # theta = ???
    # return theta


def part_1(fname_train):
    """
    This function should contain all the code you implement to complete part 1
    """
    print("========== Part 1 ==========")

    X_train, y_train = load_data(fname_train)
    Phi_train = generate_polynomial_features(X_train, 1)

    # # Example of how to use the functions
    # start = time.process_time()
    # theta = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate=0.01)
    # # theta = ls_gradient_descent(Phi_train, y_train, learning_rate=0.01)

    # print('Time elapsed:', time.process_time() - start)

    ##############################

    lrs = [1e-4, 1e-3, 1e-2, 1e-1]
    # lrs = [1e-4, 1e-3, 1e-2]

    # GD
    print("GD\n")
    for lr in lrs:
        print("=="*4 + f"learning rate:\t{lr}" + "=="*4)
        start = time.process_time()
        theta = ls_gradient_descent(Phi_train, y_train, learning_rate=lr)
        print('Time elapsed:', time.process_time() - start)
        print(f"Theta:\t{theta}\n")

    # SGD
    print("SGD\n")
    for lr in lrs:
        print("=="*4 + f"learning rate:\t{lr}" + "=="*4)
        start = time.process_time()
        theta = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate=lr)
        print('Time elapsed:', time.process_time() - start)
        print(f"Theta:\t{theta}\n")

    # Closed form
    print("=="*4 + "Closed form" + "=="*4)
    start = time.process_time()
    theta = ls_closed_form_solution(Phi_train, y_train)
    print('Time elapsed:', time.process_time() - start)
    print(f"Theta:\t{theta}\n")

    # SGD adaptive lr
    print("==" * 4 + f"learning rate:\tAdaptive" + "==" * 4)
    start = time.process_time()
    theta = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate='adaptive')
    print('Time elapsed:', time.process_time() - start)
    print(f"Theta:\t{theta}\n")

    print("Done!")


def part_2(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete part 2
    """

    def eval_gen_gap_poly_deg(X_t, y_t, X_val, y_val, M):
        poly_deg = [i for i in range(M+1)]
        train_error = []
        val_error = []
        plt.xscale("linear")

        for M in poly_deg:
            Phi_train = generate_polynomial_features(X_t, M)
            theta = ls_closed_form_solution(Phi_train, y_t)
            train_error.append(calculate_RMS_Error(Phi_train, y_t, theta))

            Phi_val = generate_polynomial_features(X_val, M)
            val_error.append(calculate_RMS_Error(Phi_val, y_val, theta))

        # Val error plot
        sns.lineplot(x=poly_deg, y=val_error, label="Validation Error")
        sns.scatterplot(x=poly_deg, y=val_error, marker='o')

        # Train error plot - we use a separate scatterplot to plot the points
        sns.lineplot(x=poly_deg, y=train_error, label="Train Error")
        sns.scatterplot(x=poly_deg, y=train_error, marker='o')

        # Add title and axis labels
        plt.title("Train vs Validation Error on Polynomial Degree")
        plt.xlabel("Polynomial Degree")
        plt.ylabel("Error")

        # Add a legend
        plt.legend()
        plt.show()

    def eval_gen_gap_reg_param(X_t, y_t, X_val, y_val, M, reg):
        poly_deg = M
        train_error = []
        val_error = []
        plt.xscale("symlog", linthresh=1e-8)

        for reg_param in reg:
            Phi_train = generate_polynomial_features(X_t, M)
            theta = ls_closed_form_solution(Phi_train, y_t, reg_param)
            train_error.append(calculate_RMS_Error(Phi_train, y_t, theta))

            Phi_val = generate_polynomial_features(X_val, M)
            val_error.append(calculate_RMS_Error(Phi_val, y_val, theta))

        # Val error plot
        sns.lineplot(x=reg, y=val_error, label="Validation Error")
        sns.scatterplot(x=reg, y=val_error, marker='o')

        # Train error plot - we use a separate scatterplot to plot the points
        sns.lineplot(x=reg, y=train_error, label="Train Error")
        sns.scatterplot(x=reg, y=train_error, marker='o')


        # Add title and axis labels
        plt.title("Train vs Validation Error on Polynomial Degree\nwith L2 Regularization")
        plt.xlabel("Regularization Parameter")
        plt.ylabel("Error")

        # Add a legend
        plt.legend()
        plt.show()

    print("=========== Part 2 ==========")

    X_train, y_train = load_data(fname_train)
    X_validation, y_validation = load_data(fname_validation)

    # TODO: Add more code here to complete part 2
    ##############################

    eval_gen_gap_poly_deg(X_train, y_train, X_validation, y_validation,
                 10)

    regs = [0] + [10**i for i in range(-8, 1)]
    print(regs)
    eval_gen_gap_reg_param(X_train, y_train, X_validation, y_validation,
                 10, regs)


    print("Done!")


def part_3(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete part 3
    """
    print("=========== Part 3 ==========")

    X_train, y_train, weights_train = load_data(fname_train, weighted=True)
    X_validation, y_validation, weights_validation = load_data(fname_validation, weighted=True)

    # TODO: Add more code here to complete part 3
    ##############################

    print("Done!")


def main(fname_train, fname_validation):
    # part_1(fname_train)
    part_2(fname_train, fname_validation)
    # part_3(fname_train, fname_validation)


if __name__ == '__main__':
    main("dataset/linreg_train.csv", "dataset/linreg_validation.csv")
