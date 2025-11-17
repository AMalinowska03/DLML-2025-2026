from collections import defaultdict

import numpy as np
import logging
from sklearn.metrics import accuracy_score


from shared.cross_validation import soft_accuracy, stratified_split
from collaborative_filtering import CollaborativeFiltering


def cross_validation(users_ratings, k_folds=6, k_neighbors=20, min_common_movies=5, tolerance=1):
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

    fold_accuracies = []
    fold_soft_accuracies = []

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

        y_true = []
        y_pred = []

        for user, movie, true_rating in val_set_tuples:
            u, m, true_rating = int(user), int(movie), int(true_rating)

            prediction = predictor.predict(u, m, k_neighbors)
            logging.info(f"Person: {u}; movie: {m}; rating: {true_rating}, prediction: {prediction}")
            y_true.append(true_rating)
            y_pred.append(prediction)

        if y_true:
            fold_acc = accuracy_score(y_true, y_pred)
            fold_soft_acc = soft_accuracy(y_true, y_pred, tolerance=tolerance)

            fold_accuracies.append(fold_acc)
            fold_soft_accuracies.append(fold_soft_acc)

    mean_accuracy = np.mean(fold_accuracies) if fold_accuracies else 0
    mean_soft_accuracy = np.mean(fold_soft_accuracies) if fold_soft_accuracies else 0
    logging.info(f"Average accuracy: {mean_accuracy}:.2f   Average soft accuracy: {mean_soft_accuracy}:.2f")

    return mean_accuracy, mean_soft_accuracy
