import numpy as np
from sklearn.model_selection import train_test_split


def train_test_split_stratified(X, y, test_size=0.2):
    # Identify rare labels (appearing only once)
    unique, counts = np.unique(y, return_counts=True)
    rare_classes = set(unique[counts == 1])

    # Separate rare samples
    mask_rare = np.isin(y, list(rare_classes))
    mask_rest = ~mask_rare
    X_rare = [row for i, row in enumerate(X) if mask_rare[i]]
    y_rare = [label for i, label in enumerate(y) if mask_rare[i]]

    X_rest = [row for i, row in enumerate(X) if mask_rest[i]]
    y_rest = [label for i, label in enumerate(y) if mask_rest[i]]

    # Stratified split only on the remaining data
    X_train, X_test, y_train, y_test = train_test_split(
        X_rest, y_rest,
        test_size=test_size,
        stratify=y_rest,
        random_state=42
    )

    # Add rare samples to training set
    if len(y_rare) > 0:
        X_train = np.concatenate([X_train, X_rare]).tolist()
        y_train = np.concatenate([y_train, y_rare], dtype=int).tolist()

    return X_train, X_test, y_train, y_test