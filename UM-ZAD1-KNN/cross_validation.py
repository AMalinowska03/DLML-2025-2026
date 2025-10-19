import numpy as np

from knn import KnnClassifier

from sklearn.metrics import accuracy_score


def cross_validation(X, y, k_folds, k_neighbors):
    accuracies = []
    folds_X = np.array_split(X, k_folds)
    folds_y = np.array_split(y, k_folds)
    for fold in range(k_folds):
        X_val = folds_X[fold]
        y_val = folds_y[fold]

        X_train = np.vstack([folds_X[i] for i in range(k_folds) if i != fold])
        y_train = np.hstack([folds_y[i] for i in range(k_folds) if i != fold])

        model = KnnClassifier(k_neighbors, X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
    return np.mean(accuracies)
