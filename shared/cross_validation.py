from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from collections import defaultdict
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
    y_pred = classifier.predict(X_val)

    return accuracy_score(y_val, y_pred), soft_accuracy(y_val, y_pred)


def cross_validation(X, y, k_folds, classifier, process_async=False):
    X, y = np.array(X), np.array(y)

    folds_X, folds_y = stratified_split(X, y, k_folds)

    accuracies = []
    soft_accuracies = []

    if process_async:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(evaluate_fold, fold, folds_X, folds_y, k_folds, classifier): fold for fold in range(k_folds)}

            for future in as_completed(futures):
                acc, soft_acc = future.result()
                accuracies.append(acc)
                soft_accuracies.append(soft_acc)

    else:
        for fold in range(k_folds):
            acc, soft_acc = evaluate_fold(fold, folds_X, folds_y, k_folds, classifier)
            accuracies.append(acc)
            soft_accuracies.append(soft_acc)

    return np.mean(accuracies).tolist(), np.mean(soft_accuracies).tolist()
