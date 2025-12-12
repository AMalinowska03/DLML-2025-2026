from collections import defaultdict

import numpy as np
import logging
from sklearn.metrics import accuracy_score

from shared.cross_validation import soft_accuracy, stratified_split
from collaborative_filtering import CollaborativeFiltering


def cross_validation(users_ratings, k_folds=20, min_common_movies=5):
    """
    Przeprowadza pełną k-krotną walidację krzyżową dla modelu CF.

    Args:
        all_ratings_tuples (list): Lista krotek (user_id, movie_id, rating)
        k_folds (int): Liczba foldów CV.
        k_neighbors (int): Hiperparametr K (liczba sąsiadów) do testowania.
        min_common_movies (int): Hiperparametr dla liczenia podobieństwa.
        tolerance (int): Tolerancja dla soft_accuracy.

    Returns:
        tuple: (mean_accuracy, mean_soft_accuracy)
    """

    all_ratings_tuples = []
    for user_id, ratings in users_ratings.items():
        for movie_id, rating in ratings.items():
            all_ratings_tuples.append((user_id, movie_id, rating))

    y_labels = [r for u, m, r in all_ratings_tuples]

    folds_X, _ = stratified_split(np.array(all_ratings_tuples), np.array(y_labels), k_folds)

    train_accuracies = []
    train_soft_accuracies = []
    val_accuracies = []
    val_soft_accuracies = []

    for i in range(k_folds):
        val_set_tuples = folds_X[i]

        train_set_list = []
        for j in range(k_folds):
            if i != j:
                train_set_list.extend(folds_X[j])

        temp_user_ratings = defaultdict(dict)
        for user, movie, rating in train_set_list:
            temp_user_ratings[int(user)][int(movie)] = float(rating)

        if not temp_user_ratings:
            continue  # empty

        predictor = CollaborativeFiltering(users_ratings=temp_user_ratings)
        predictor.calculate_average_ratings()
        predictor.calculate_similarities(min_common_movies)

        train_accuracy, train_soft_accuracy = _calculate_accuracy(predictor, train_set_list)
        val_accuracy, val_soft_accuracy = _calculate_accuracy(predictor, val_set_tuples)

        train_accuracies.append(train_accuracy)
        train_soft_accuracies.append(train_soft_accuracy)
        val_accuracies.append(val_accuracy)
        val_soft_accuracies.append(val_soft_accuracy)

    mean_train_accuracy = np.mean(train_accuracies) if train_accuracies else 0
    mean_train_soft_accuracy = np.mean(train_soft_accuracies) if train_soft_accuracies else 0
    mean_val_accuracy = np.mean(val_accuracies) if val_accuracies else 0
    mean_val_soft_accuracy = np.mean(val_soft_accuracies) if val_soft_accuracies else 0

    logging.info(f"TRAIN - Average accuracy: {mean_train_accuracy:.4f}   Average soft accuracy: {mean_train_soft_accuracy:.4f}")
    logging.info(f"VALIDATE - Average accuracy: {mean_val_accuracy:.4f}   Average soft accuracy: {mean_val_soft_accuracy:.4f}")


def _calculate_accuracy(classifier, val_set_tuples):
    y_true = []
    y_pred = []
    for user, movie, true_rating in val_set_tuples:
        u, m, true_rating = int(user), int(movie), int(true_rating)

        prediction = classifier.predict(u, m)
        y_true.append(true_rating)
        y_pred.append(prediction)

    return accuracy_score(y_true, y_pred), soft_accuracy(y_true, y_pred)
