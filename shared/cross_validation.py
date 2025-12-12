from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score

def soft_accuracy(y_true, y_pred, tolerance=1):
    correct = sum(abs(t - p) <= tolerance for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def stratified_split(X, y, k_folds):
    label_indices = defaultdict(list)
    for i, label in enumerate(y):
        label_indices[label].append(i)

    for label in label_indices:
        np.random.shuffle(label_indices[label])

    folds = [[] for _ in range(k_folds)]
    for label, indices in label_indices.items():
        for i, idx in enumerate(indices):
            folds[i % k_folds].append(idx)

    folds_X = [X[fold] for fold in folds]
    folds_y = [y[fold] for fold in folds]

    return folds_X, folds_y


def evaluate_fold(fold, folds_X, folds_y, k_folds, classifier):
    X_val = folds_X[fold]
    y_val = folds_y[fold]

    X_train = np.concatenate([folds_X[i] for i in range(k_folds) if i != fold])
    y_train = np.concatenate([folds_y[i] for i in range(k_folds) if i != fold])

    classifier.fit(X_train, y_train)
    if hasattr(classifier, "is_forest") and not classifier.is_forest:
        classifier.prune(X_val, y_val)
    y_val_pred = classifier.predict(X_val)
    y_train_pred = classifier.predict(X_train)

    return (accuracy_score(y_train, y_train_pred), soft_accuracy(y_train, y_train_pred),
            accuracy_score(y_val, y_val_pred), soft_accuracy(y_val, y_val_pred))


def cross_validation(X, y, k_folds, classifier):
    X, y = np.array(X), np.array(y)

    folds_X, folds_y = stratified_split(X, y, k_folds)

    train_accuracies = []
    train_soft_accuracies = []
    val_accuracies = []
    val_soft_accuracies = []
    for fold in range(k_folds):
        train_acc, train_soft_acc, val_acc, val_soft_acc = evaluate_fold(fold, folds_X, folds_y, k_folds, classifier)
        train_accuracies.append(train_acc)
        train_soft_accuracies.append(train_soft_acc)
        val_accuracies.append(val_acc)
        val_soft_accuracies.append(val_soft_acc)

    return np.mean(train_accuracies), np.mean(train_soft_accuracies), np.mean(val_accuracies), np.mean(val_soft_accuracies)
