"""
Ensemble Methods
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import random
from logging import critical

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

import sklearn.tree
from sklearn import metrics, utils
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def majority_vote(y_preds):
    """
    Given a list of m-many (n,) arrays, with each (n,) array containing the predictions for
    the same set of n examples (using different classifiers), return a (n,) array
    containing the majority vote predictions of the m-many (n,) arrays. If tie occurs,
    randomly choose a label of the plurality labels.
    Input:
        y_preds : list (m) - a list of m (n,) arrays
    Returns:
        y_pred : np.array (n) - array containing majority vote predictions
    """
    y_pred = []
    for y in np.array(y_preds).T:
        c = Counter(y).most_common()
        c = [i for i,n in c if n == c[0][1]]
        y_pred.append(random.choice(c))
    return np.array(y_pred)


def random_forest(X_train, y_train, X_test, y_test, m, n_clf):
    """
    Returns accuracy on the test set X_test with corresponding labels y_test
    using a random forest classifier with n_clf decision trees trained with
    training examples X_train and training labels y_train.
    Input:
        X_train : np.array (n_train, d) - array of training feature vectors
        y_train : np.array (n_train) - array of labels corresponding to X_train samples
        X_test : np.array (n_test,d) - array of testing feature vectors
        y_test : np.array (n_test) - array of labels corresponding to X_test samples
        n_clf : int - number of decision tree classifiers in the random forest
    Returns:
        accuracy : float - accuracy of random forest classifier on X_test samples
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # res = np.zeros((n_test, n_clf))
    bres = []

    for i in range(n_clf):
        indices = np.random.choice(n_train, n_train, replace=True)
        X_bs = X_train[indices]
        y_bs = y_train[indices]


        clf = DecisionTreeClassifier(criterion="entropy", max_features=m)
        clf.fit(X_bs, y_bs)
        preds = clf.predict(X_test)

        # res[:, i] = preds
        bres.append(preds)

    # final_pred = np.zeros(n_test, dtype=int)
    #
    # for i in range(n_test):
    #     vals, cnts = np.unique(res[i, :], return_counts=True)
    #     max_cnt = np.max(cnts)
    #     cands = vals[cnts == max_cnt]
    #     final_pred[i] = np.random.choice(cands)
    #
    fpred = majority_vote(bres)

    n_correct = 0

    for i in range(n_test):
        # if final_pred[i] == y_test[i]:
        #     n_correct += 1
        if fpred[i] == y_test[i]:
            n_correct += 1

    accuracy = n_correct/n_test

    return accuracy


def bagging_ensemble(X_train, y_train, X_test, y_test, n_clf):
    """
    Returns accuracy on the test set X_test with corresponding labels y_test
    using a bagging ensemble classifier with n_clf decision trees trained with
    training examples X_train and training labels y_train.
    Input:
        X_train : np.array (n_train, d) - array of training feature vectors
        y_train : np.array (n_train) - array of labels corresponding to X_train samples
        X_test : np.array (n_test,d) - array of testing feature vectors
        y_test : np.array (n_test) - array of labels corresponding to X_test samples
        n_clf : int - number of decision tree classifiers in the random forest, default is 10
    Returns:
        accuracy : float - accuracy of random forest classifier on X_test samples
    """
    return random_forest(X_train, y_train, X_test, y_test, m=None, n_clf=n_clf)


def load_mnist(classes):
    """
    Load MNIST dataset for classes
    Every 25th sample is used to reduce computational resources
    Input:
        classes : list of ints
    Returns:
        X : np.array (num_samples, num_features)
        y : np.array (num_samples)
    """
    print('Fetching MNIST data...')
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X_all = np.array(mnist.data)[::25]  # Every 25th sample is used to reduce computational requirements
    y_all = np.array(mnist.target)[::25]
    X_all = X_all.reshape(-1, 28, 28).reshape(-1, 14, 2, 14, 2).mean(axis=(2,4)).reshape(-1, 14*14)  # downsample image to 14x14
    y_all = y_all.astype(int)  # convert string labels to integers
    desired_idx = np.isin(y_all, classes)
    return X_all[desired_idx], y_all[desired_idx]


def main():
    """
    Analyze how the performance of bagging and random forest changes with m.
    """
    # Use a subset of MNIST data
    X, y = load_mnist([1,3,5])

    # Get average performance of bagging and random forest with different m values
    print('Getting bagging and random forest scores...')
    m_vals = [1, 2, 5, 10, 20, 40, 80, 120, 160, 196]
    rf_avg = []
    bag_avg = []
    for m in m_vals:
        print('m = {}'.format(m))
        bagging_scores = []
        random_forest_scores = []
        for i in range(50):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            random_forest_scores.append(random_forest(X_train, y_train, X_test, y_test, m, n_clf=10))
            bagging_scores.append(bagging_ensemble(X_train, y_train, X_test, y_test, n_clf=10))
        rf_avg.append(np.mean(np.array(random_forest_scores)))
        bag_avg.append(np.mean(np.array(bagging_scores)))
    
    # Plot accuracies against m
    plt.figure()
    plt.plot(list(m_vals), bag_avg, '--', label='bagging')
    plt.plot(list(m_vals), rf_avg, '--', label='random forest')
    plt.xlabel('m')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.savefig('ensemble.png')


if __name__ == '__main__':
    np.random.seed(69)
    main()
    # X, y = load_mnist([1,3,5])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # random_forest(X_train, y_train, X_test, y_test, 10, 10)