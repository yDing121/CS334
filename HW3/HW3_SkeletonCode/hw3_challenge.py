import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import random


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import hw3_main
from helper import *

random.seed(42)

def generate_feature_vector_challenge(df):
    static_variables = config['static']
    timeseries_variables = config['timeseries']
    feature_dict = {}

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
            fval = np.nan
            sval = np.nan
            norm_sd = np.nan
        else:
            fval = (df[(df["Variable"] == var) & (df["Time"].str[:2].astype(int) < 24)]["Value"]).mean()
            sval = (df[(df["Variable"] == var) & (df["Time"].str[:2].astype(int) >= 24)]["Value"]).mean()

            mean = (df[(df["Variable"] == var)]["Value"]).mean()
            if mean == 0:
                norm_sd = np.nan
            else:
                sd = (df[(df["Variable"] == var)]["Value"]).std()
                norm_sd = sd / mean

        feature_dict[f"f24_mean_{var}"] = fval
        feature_dict[f"s24_mean_{var}"] = sval
        feature_dict[f"norm_sd_{var}"] = norm_sd

    return feature_dict

def normalize_feature_matrix_challenge(X):
    return hw3_main.normalize_feature_matrix(X)

def impute_missing_values_challenge(X):
    for col in range(X.shape[1]):
        median = np.nanmedian(X[:, col])
        X[:, col] = np.where(np.isnan(X[:, col]), median, X[:, col])

    return X

def get_train_val_split(X: np.ndarray[float], y: np.ndarray[int]):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y, random_state=69)
    return X_train, X_val, y_train, y_val

def run_challenge(X_challenge, y_challenge, X_heldout, feature_names):
    print("================= Part 3 ===================")
    print("Part 3: Challenge")

    X_train, X_val, y_train, y_val = get_train_val_split(X_challenge, y_challenge)
    assert X_train.shape[0] == y_train.size and X_val.shape[0] == y_val.size

    alpha_range = np.logspace(-4, 4, 9)
    penalties = ["l1", "l2", "elasticnet"]
    scores = []

    for alpha in alpha_range:
        for penalty in penalties:
            clf = SGDClassifier(loss="modified_huber", alpha=alpha, penalty=penalty)
            clf.fit(X_train, y_train)

            score = hw3_main.cv_performance(clf, X_train, y_train, 10, "auroc")
            print("alpha: {:.6f} \t penalty: {:10s} \t score: {:.4f}".format(alpha, penalty, score))
            scores.append((alpha, penalty, score))

    best = sorted(scores, key=lambda x: x[2], reverse=True)[0]
    clf = SGDClassifier(loss="modified_huber", alpha=best[0], penalty=best[1])
    clf.fit(X_train, y_train)

    test_perf = hw3_main.performance(clf, X_val, y_val, "auroc")
    print("alpha = " + str(best[0]) + "\npenalty = " + str(best[1]) +
          "\nTest Performance on metric " + "auroc" + ": %.4f" % test_perf)

    metric_list = ["accuracy", "precision", "sensitivity", "specificity", "f1_score", "auroc", "auprc"]

    for metric in metric_list:
        test_perf = hw3_main.performance(clf, X_val, y_val, metric)
        print("Validation Performance on metric " + metric + ": %.4f" % test_perf)

    y_score = clf.predict_proba(X_heldout)[:, 1]
    y_label = clf.predict(X_heldout)
    make_challenge_submission(y_label, y_score)


if __name__ == '__main__':
    # Read challenge data
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()

    # TODO: Question 3: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    run_challenge(X_challenge, y_challenge, X_heldout, feature_names)
    test_challenge_output()
