import logging
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

from cross_validation import cross_validation
from knn import KnnClassifier

default_k = 7
cross_validation_folds = 10


def _evaluate_single_k(X_scaled, y, k, person):
    acc, soft_acc = cross_validation(X_scaled, y, cross_validation_folds, k)
    logging.debug(f"Accuracy for person {person} with k={k}: acc: {acc}, soft_acc: {soft_acc}")
    return k, acc, soft_acc


def train_classifier(person, encoded_features, person_ratings, find_best_k=False):
    logging.info(f"Training classifier for person {person}...")

    X = [encoded_features[movie_number] for movie_number in person_ratings]
    y = [person_ratings[movie_number] for movie_number in person_ratings]
    # X_scaled = MaxAbsScaler().fit_transform(X)

    accuracy = 0
    soft_accuracy = 0
    k_neighbours = default_k

    if find_best_k:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_evaluate_single_k, X, y, k, person): k for k in range(5, 20)}

            for future in as_completed(futures):
                k, acc, soft_acc = future.result()
                if acc > accuracy:
                    accuracy = acc
                    k_neighbours = k
                    soft_accuracy = soft_acc
    else:
        accuracy, soft_accuracy = cross_validation(X, y, cross_validation_folds, k_neighbours)

    logging.info(f"Accuracy for person {person} with k={k_neighbours}: acc: {accuracy}, soft_acc: {soft_accuracy}")

    return accuracy, soft_accuracy, KnnClassifier(k=k_neighbours, data=X, data_labels=y)
