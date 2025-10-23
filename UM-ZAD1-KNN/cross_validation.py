import numpy as np

from knn import KnnClassifier

from sklearn.metrics import accuracy_score

n_jobs = 4

def evaluate_fold(fold, folds_X, folds_y, k_folds, k_neighbors):
    X_val = folds_X[fold]
    y_val = folds_y[fold]

    X_train = []
    for i in range(k_folds):
        if i != fold:
            X_train.extend(folds_X[i])

    y_train = []
    for i in range(k_folds):
        if i != fold:
            y_train.extend(folds_y[i])
    # X_train = np.vstack([folds_X[i] for i in range(k_folds) if i != fold])
    # y_train = np.hstack([folds_y[i] for i in range(k_folds) if i != fold])

    model = KnnClassifier(k_neighbors, X_train, y_train)
    y_pred = model.predict(X_val)

    return accuracy_score(y_val, y_pred), soft_accuracy(y_val, y_pred)


def cross_validation(X, y, k_folds, k_neighbors):
    folds_X = np.array_split(X, k_folds)
    folds_y = np.array_split(y, k_folds)

    accuracies = []
    soft_accuracies = []
    for fold in range(k_folds):
        acc, soft_acc = evaluate_fold(fold, folds_X, folds_y, k_folds, k_neighbors)
        accuracies.append(acc)
        soft_accuracies.append(soft_acc)

    return np.mean(accuracies).tolist(), np.mean(soft_accuracies).tolist()


def soft_accuracy(y_true, y_pred, tolerance=1):
    correct = sum(abs(t - p) <= tolerance for t, p in zip(y_true, y_pred))
    return correct / len(y_true)
