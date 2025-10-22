import logging
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

from cross_validation import cross_validation
from knn import KnnClassifier

default_k = 7
cross_validation_folds = 10


def _evaluate_single_k(X_scaled, y, k, person):
    acc = cross_validation(X_scaled, y, cross_validation_folds, k)
    logging.info(f"Accuracy for person {person} with k={k}: {acc}")
    return k, acc


def train_classifier(person, encoded_features, person_ratings, find_best_k=False):
    logging.info(f"Training classifier for person {person}...")

    X = [np.concatenate(list(encoded_features[movie_number].values())) for movie_number in person_ratings]
    y = [person_ratings[movie_number] for movie_number in person_ratings]
    X_scaled = MaxAbsScaler().fit_transform(X)

    accuracy = 0
    k_neighbours = default_k

    if find_best_k:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_evaluate_single_k, X_scaled, y, k, person): k for k in range(5, 20)}

            for future in as_completed(futures):
                k, acc = future.result()
                if acc > accuracy:
                    accuracy = acc
                    k_neighbours = k
    else:
        accuracy = cross_validation(X_scaled, y, cross_validation_folds, k_neighbours)
        logging.info(f"Accuracy for person {person} with k={k_neighbours}: {accuracy}")

    return accuracy, KnnClassifier(k=k_neighbours, data=X_scaled, data_labels=y)
