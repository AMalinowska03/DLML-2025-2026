import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from shared.cross_validation import cross_validation
from knn import KnnClassifier

default_k = 7
cross_validation_folds = 10


def _evaluate_single_k(X_scaled, y, k, person):
    classifier = KnnClassifier(k=k)
    train_acc, train_soft_acc, val_acc, val_soft_acc = cross_validation(X_scaled, y, cross_validation_folds, classifier)
    logging.debug(f"TRAIN - Accuracy for person {person} with k={k}: acc: {train_acc}, soft_acc: {train_soft_acc}")
    logging.debug(f"VALIDATE - Accuracy for person {person} with k={k}: acc: {val_acc}, soft_acc: {val_soft_acc}")
    return k, train_acc, train_soft_acc, val_acc, val_soft_acc


def train_classifier(person, encoded_features, person_ratings, find_best_k=False):
    logging.info(f"Training classifier for person {person}...")

    X = [encoded_features[movie_number] for movie_number in person_ratings]
    y = [person_ratings[movie_number] for movie_number in person_ratings]

    train_accuracy = 0
    train_soft_accuracy = 0
    val_accuracy = 0
    val_soft_accuracy = 0
    k_neighbours = default_k

    if find_best_k:
        with ThreadPoolExecutor(1) as executor:
            futures = {executor.submit(_evaluate_single_k, X, y, k, person): k for k in range(5, 20)}

            for future in as_completed(futures):
                k, train_acc, train_soft_acc, val_acc, val_soft_acc = future.result()
                if val_acc > val_accuracy:
                    val_accuracy = val_acc
                    val_soft_accuracy = val_soft_acc
                    train_accuracy = train_acc
                    train_soft_accuracy = train_soft_acc
                    k_neighbours = k
    else:
        classifier = KnnClassifier(k=k_neighbours)
        train_accuracy, train_soft_accuracy, val_accuracy, val_soft_accuracy = cross_validation(X, y, cross_validation_folds, classifier)

    logging.info(f"Accuracy for person {person} with k={k_neighbours}: "
                 f"train_acc: {train_accuracy}, train_soft_acc: {train_soft_accuracy} "
                 f"val_acc: {val_accuracy}, val_soft_acc: {val_soft_accuracy}")

    classifier = KnnClassifier(k=k_neighbours)
    classifier.fit(X, y)

    return train_accuracy, train_soft_accuracy, val_accuracy, val_soft_accuracy, classifier
