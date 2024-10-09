# hw3_main.py

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, exceptions

import warnings
warnings.filterwarnings('ignore', category=exceptions.UndefinedMetricWarning)

from helper import *


def generate_feature_vector(df):
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Input:
        df: pd.Dataframe, with columns [Time, Variable, Value]

    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {'Age': 32, 'Gender': 0, 'mean_HR': 84, ...}
    """
    static_variables = config['static']
    timeseries_variables = config['timeseries']
    feature_dict = {}

    # Static variables
    for var in static_variables:
        val = df[df["Variable"] == var]["Value"].values[0]
        feature_dict[var] = val

        if val < 0:
            feature_dict[var] = np.nan
        else:
            feature_dict[var] = val

    # Time-varying variables
    for var in timeseries_variables:
        if (df['Variable'] == var).sum() == 0:
            val = np.nan
        else:
            val = (df[df["Variable"] == var]["Value"]).mean()
        feature_dict[f"mean_{var}"] = val


    return feature_dict


def impute_missing_values(X):
    """
    For each feature column, impute missing values  (np.nan) with the
    population mean for that feature.

    Input:
        X: np.array, shape (N, d). X could contain missing values
    Returns:
        X: np.array, shape (N, d). X does not contain any missing values
    """


    for col in range(X.shape[1]):
        mean = np.nanmean(X[:, col])
        X[:, col] = np.where(np.isnan(X[:, col]), mean, X[:, col])

    return X


# def normalize_feature_matrix(X):
#     """
#     For each feature column, normalize all values to range [0, 1].
#
#     Input:
#         X: np.array, shape (N, d).
#     Returns:
#         X: np.array, shape (N, d). Values are normalized per column.
#     """
#     # TODO: Implement this function
#     ???
#     return X
#
#
# def performance(clf, X, y_true, metric='accuracy'):
#     """
#     Calculates the performance metric as evaluated on the true labels
#     y_true versus the predicted scores from clf and X.
#     Input:
#         clf: an instance of sklearn estimator
#         X : (N,d) np.array containing features
#         y_true: (N,) np.array containing true labels
#         metric: string specifying the performance metric (default='accuracy'
#                  other options: 'precision', 'sensitivity', 'specificity',
#                 'f1-score', 'auroc', and 'auprc')
#     Returns:
#         the performance measure as a float
#     """
#     # TODO: Implement this function
#     ???
#
#
# def cv_performance(clf, X, y, k=5, metric='accuracy'):
#     """
#     Splits the data X and the labels y into k folds.
#     Then, for each fold i in 1...k,
#         Train a classifier on all the data except the i-th fold, and test on the i-th fold.
#         Calculate the performance of the classifier and save the result.
#     In the end, return the average performance across the k folds.
#     Input:
#         clf: an instance of sklearn estimator
#         X: (N,d) array of feature vectors, where n is the number of examples
#            and d is the number of features
#         y: (N,) array of binary labels {1,-1}
#         k: an int specifying the number of folds (default=5)
#         metric: string specifying the performance metric (default='accuracy'
#              other options: 'precision', 'sensitivity', 'specificity',
#                 'f1-score', 'auroc', and 'auprc')
#     Returns:
#         average cross-validation performance across the k folds as a float
#     """
#     # TODO: Implement this function
#     skf = StratifiedKFold(n_splits=k)
#     scores = []
#     # For each split in the k folds...
#     for train, val in skf.split(X,y):
#         # Split the data into training and validation sets...
#         X_train, y_train, X_val, y_val = ???
#         # Fit the data to the training data...
#         ???
#         # And test on the ith fold.
#         score = ???
#         scores.append(score)
#     # Return the average performance across all fold splits.
#     return np.array(scores).mean()
#
#
# def select_C(X, y, C_range=[], penalty='l2', k=5, metric='accuracy'):
#     """
#     Sweeps different C hyperparameters of a logistic regression classifier,
#     calculates the k-fold CV performance for each setting on dataset (X, y),
#     and return the best C.
#     Input:
#         X: (N,d) array of feature vectors, where N is the number of examples
#             and d is the number of features
#         y: (N,) array of binary labels {1,-1}
#         k: int specifying the number of folds for cross-validation (default=5)
#         metric: string specifying the performance metric (default='accuracy',
#              other options: 'precision', 'sensitivity', 'specificity',
#                 'f1-score', 'auroc', and 'auprc')
#         penalty: whether to use 'l1' or 'l2' regularization (default='l2')
#         C_range: a list of hyperparameter C values to be searched over
#     Returns:
#         the C value for a logistic regression classifier that maximizes
#         the average 5-fold CV performance.
#     """
#     print("{}-regularized Logistic Regression "
#           "Hyperparameter Selection based on {}:".format(penalty.upper(), metric))
#     scores = []
#     # Iterate over all of the given C range...
#     for C in C_range:
#         # Calculate the average performance on k-fold cross-validation
#         clf = get_classifier(penalty=penalty, C=C)
#         score = cv_performance(clf, X, y, k, metric)
#         print("C: {:.6f} \t score: {:.4f}".format(C, score))
#         scores.append((C, score))
#     # Return the C value with the maximum score
#     maxval = max(scores, key=lambda x: x[1])
#     return maxval[0]
#
#
# def plot_coefficients(X, y, penalty, C_range):
#     """
#     Takes as input the training data X and labels y and plots the L0-norm
#     (number of nonzero elements) of the coefficients learned by a classifier
#     as a function of the C-values of the classifier.
#     """
#     print("Plotting the number of nonzero entries of the parameter vector as a function of C")
#     norm0 = []
#
#     # TODO: Implement this function
#     ???
#
#     # This code will plot your L0-norm as a function of C
#     plt.plot(C_range, norm0)
#     plt.axhline(y=X.shape[1], color='gray', linestyle=':')
#     plt.xscale('log')
#     plt.legend(['L0-norm'])
#     plt.xlabel("Value of C")
#     plt.ylabel("L0-norm of theta")
#     plt.ylim(-2,50)
#     plt.title('L0-norm of θ vs C, {}-penalized logistic regression'.format(penalty.upper()))
#     plt.savefig('l0-norm_vs_C__'+penalty+'-penalty.png', dpi=200)
#     plt.close()
#
#     print('Plot saved')
#
#
# def q1(X, feature_names):
#     """
#     Given a feature matrix X, prints d, the number of features in the feature vector,
#     and prints the average feature vector along with its corresponing feature name.
#     """
#     ##################################################################
#     print("--------------------------------------------")
#     print("Question 1(d): reporting dataset statistics:")
#     print("\t", "d:", X.shape[1])
#     print("\t", "Average feature vector:")
#     print(pd.DataFrame({"Feature Name": feature_names, "Mean value": X.mean(axis=0)}))
#
#
# def q2(X_train, y_train, X_test, y_test, metric_list, feature_names):
#     """
#     This function should contain all the code you implement to complete part 2
#     """
#     print("================= Part 2 ===================")
#
#     C_range = np.logspace(-3, 3, 7)
#
#     ##################################################################
#     print("--------------------------------------------")
#     print("Question 2.1(c): Logistic Regression with L2-penalty, grid search, all metrics")
#     for metric in metric_list:
#         best_C = select_C(X_train, y_train, C_range, 'l2', 5, metric)
#         print("Best C: %.6f" % best_C)
#
#     ##################################################################
#     print("--------------------------------------------")
#     print("Question 2.1(d): Test Performance of L2-reg logistic regression with best C")
#     best_C = ???
#     ??? # Fit the classifier with the best C
#     for metric in metric_list:
#         test_perf = performance(clf, X_test, y_test, metric)
#         print("C = " + str(best_C) + " Test Performance on metric " + metric + ": %.4f" % test_perf)
#
#     ##################################################################
#     print("--------------------------------------------")
#     print("Question 2.1(e): Plot L0-norm of theta coefficients vs. C, l2 penalty")
#     plot_coefficients(X_train, y_train, 'l2', C_range)
#
#     ##################################################################
#     print("--------------------------------------------")
#     print("Question 2.1(f): Displaying the most positive and negative coefficients and features")
#     best_C = ???
#     print('Positive coefficients...')
#     print('Negative coefficients...')
#
#
#     ##################################################################
#     print("--------------------------------------------")
#     print("Question 2.2(a): Logistic Regression with L1-penalty, grid search, AUROC")
#     best_C = ???
#     test_performance = ???
#
#     ##################################################################
#     print("--------------------------------------------")
#     print("Question 2.2(b): Plot the weights of C vs. L0-norm of theta, l1 penalty")
#     plot_coefficients(X_train, y_train, 'l1', C_range)
#
#
# def main():
#     np.random.seed(42)
#
#     # Read data
#     # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
#     #       IMPLEMENTING generate_feature_vector, fill_missing_values AND normalize_feature_matrix
#     X_train, y_train, X_test, y_test, feature_names = get_train_test_split()
#
#     # TODO: Questions 1, 2
#     metric_list = ["accuracy", "precision", "sensitivity", "specificity", "f1_score", "auroc", "auprc"]
#
#     q1(X_train, feature_names)
#     q2(X_train, y_train, X_test, y_test, metric_list, feature_names)


if __name__ == '__main__':
    # main()
    df_labels = pd.read_csv('data/labels.csv')
    df_labels = df_labels[:2500]
    IDs = df_labels['RecordID'][:2500]
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv('data/files/{}.csv'.format(i))

    features = Parallel(n_jobs=16)(delayed(hw3_main.generate_feature_vector)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features).sort_index(axis=1)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['In-hospital_death'].values

    # print(X)

    X = hw3_main.impute_missing_values(X)