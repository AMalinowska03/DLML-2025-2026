from collections import defaultdict

import numpy as np
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

    train_accuracy, train_soft_accuracy = _calculate_accuracy(classifier, X_train, y_train)
    val_accuracy, val_soft_accuracy = _calculate_accuracy(classifier, X_val, y_val)

    return train_accuracy, train_soft_accuracy, val_accuracy, val_soft_accuracy


def _calculate_accuracy(classifier, X, y):
    y_true = []
    y_pred = []
    for index, (user, movie) in enumerate(X):
        u, m, true_rating = int(user), int(movie), int(y[index])

        prediction = classifier.predict(u, m)
        y_true.append(true_rating)
        y_pred.append(prediction)

    return accuracy_score(y_true, y_pred), soft_accuracy(y_true, y_pred)


def cross_validation(users_ratings, k_folds=15):
    X = []
    y = []
    for user_id, ratings in users_ratings.items():
        for movie_id, rating in ratings.items():
            X.append((user_id, movie_id))
            y.append(rating)

    folds_X, folds_y = stratified_split(np.array(X), np.array(y), k_folds)

    train_accuracies = []
    train_soft_accuracies = []
    val_accuracies = []
    val_soft_accuracies = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_fold, fold, folds_X, folds_y) for fold in range(k_folds)]

        for future in as_completed(futures):
            train_acc, train_soft_acc, val_acc, val_soft_acc = future.result()
            train_accuracies.append(train_acc)
            train_soft_accuracies.append(train_soft_acc)
            val_accuracies.append(val_acc)
            val_soft_accuracies.append(val_soft_acc)

    return np.mean(train_accuracies).tolist(), np.mean(train_soft_accuracies).tolist(), np.mean(val_accuracies).tolist(), np.mean(val_soft_accuracies).tolist()


def compute_rmse(y_true, y_pred):
    error = []
    for i in range(len(y_true)):
        error.append((y_true[i] - y_pred[i]) ** 2)
    return np.sqrt(np.mean(error))