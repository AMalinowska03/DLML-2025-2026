from collections import defaultdict

import numpy as np
import logging
from sklearn.metrics import accuracy_score


from shared.cross_validation import soft_accuracy, stratified_split
from collaborative_filtering import CollaborativeFiltering


from concurrent.futures import ProcessPoolExecutor, as_completed


def evaluate_fold(fold, folds_X, folds_y):
    k_folds = len(folds_X)

    X_val = folds_X[fold]
    y_val = folds_y[fold]

    X_train = np.concatenate([folds_X[i] for i in range(k_folds) if i != fold])
    y_train = np.concatenate([folds_y[i] for i in range(k_folds) if i != fold])

    temp_user_ratings = defaultdict(dict)
    for index, (user, movie) in enumerate(X_train):
        temp_user_ratings[int(user)][int(movie)] = float(y_train[index])

    classifier = CollaborativeFiltering(users_ratings=temp_user_ratings)
    classifier.train()

    y_true = []
    y_pred = []
    for index, (user, movie) in enumerate(X_val):
        u, m, true_rating = int(user), int(movie), int(y_val[index])

        prediction = classifier.predict(u, m)
        logging.info(f"Person: {u}; movie: {m}; rating: {true_rating}, prediction: {prediction}")
        y_true.append(true_rating)
        y_pred.append(prediction)

    return accuracy_score(y_true, y_pred), soft_accuracy(y_true, y_pred)


def cross_validation(users_ratings, k_folds=20):
    X = []
    y = []
    for user_id, ratings in users_ratings.items():
        for movie_id, rating in ratings.items():
            X.append((user_id, movie_id))
            y.append(rating)

    folds_X, folds_y = stratified_split(np.array(X), np.array(y), k_folds)

    accuracies = []
    soft_accuracies = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_fold, fold, folds_X, folds_y) for fold in range(k_folds)]

        for future in as_completed(futures):
            acc, soft_acc = future.result()
            accuracies.append(acc)
            soft_accuracies.append(soft_acc)

    return np.mean(accuracies).tolist(), np.mean(soft_accuracies).tolist()
