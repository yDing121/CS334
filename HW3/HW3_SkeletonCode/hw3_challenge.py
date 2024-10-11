# hw3_challenge.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression

import hw3_main
from helper import *

def generate_feature_vector_challenge(df):
    return hw3_main.generate_feature_vector(df)

def impute_missing_values_challenge(X):
    return hw3_main.impute_missing_values(X)

def normalize_feature_matrix_challenge(X):
    return hw3_main.normalize_feature_matrix(X)

def get_train_val_split(X: np.ndarray[float], y: np.ndarray[int]):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y, random_state=69)
    return X_train, X_val, y_train, y_val

def preprocess(X: np.ndarray[float], feature_names):
    tmp = dict()

    for i, k in enumerate(feature_names):
        tmp[k] = X[:, i]

    df = pd.DataFrame(tmp)
    print(df.head())

def run_challenge(X_challenge, y_challenge, X_heldout, feature_names):
    # TODO:
    # Read challenge data
    # Train a linear classifier and apply to heldout dataset features
    # Use generate_challenge_labels to print the predicted labels
    print("================= Part 3 ===================")
    print("Part 3: Challenge")
    clf = LogisticRegression() ### TODO: define your classifier with appropriate hyperparameters
    clf.fit(X_challenge, y_challenge)

    X_train, X_val, y_train, y_val = get_train_val_split(X_challenge, y_challenge)
    assert X_train.shape[0] == y_train.size and X_val.shape[0] == y_val.size


    C_range = np.logspace(-4, 4, 9)




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
