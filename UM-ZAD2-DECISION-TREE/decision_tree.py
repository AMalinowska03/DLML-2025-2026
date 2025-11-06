from collections import Counter
import random

import numpy as np
from fontTools.varLib.avar.plan import measureSlant


def is_list_of_lists(values):
    return isinstance(values, list) and all(isinstance(v, list) for v in values)

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf value


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10, information_gain='gini', features_number_to_compare=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = 0.01
        self.features_number_to_compare = features_number_to_compare
        self.root = None
        self.information_gain = information_gain

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self.predict_single(row) for row in X])

    def predict_single(self, row):
        node = self.root
        while node.value is None:
            val = row[node.feature]
            if isinstance(val, list):
                go_left = node.threshold in val
            else:
                if isinstance(node.threshold, (int, float, np.number)):
                    go_left = val <= node.threshold
                else:
                    go_left = val == node.threshold
            node = node.left if go_left else node.right
        return node.value

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = len(X), len(X[0])
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)

        feature, thresh = self._best_split(X, y)
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)

        X_left, X_right, y_left, y_right = self._split(X, y, feature, thresh)

        left = self._grow_tree(X_left, y_left, depth + 1)
        right = self._grow_tree(X_right, y_right, depth + 1)
        return DecisionNode(feature, thresh, left, right)

    def _best_split(self, X, y):
        best_feature, best_threshold, best_gain = None, None, 0
        base_measure = self._measure(y)

        all_features = X[0].keys()
        if self.features_number_to_compare is None:
            chosen_features = all_features
        else:
            feature_number = min(self.features_number_to_compare, len(all_features))
            chosen_features = random.sample(list(all_features), feature_number)

        for feature in chosen_features:
            unique_values = self.get_unique_values(X, feature)

            for value in unique_values:
                _, _, y_left, y_right = self._split(X, y, feature, value)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                measure_left = self._measure(y_left)
                measure_right = self._measure(y_right)
                weighted_measure = (len(y_left) * measure_left + len(y_right) * measure_right) / len(y)
                gain = base_measure - weighted_measure

                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, value, gain

        if best_gain < self.min_gain:
            return None, None

        return best_feature, best_threshold

    def _measure(self, y):
        if self.information_gain == 'gini':
            return self._gini(y)
        elif self.information_gain == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown measure: {self.information_gain}. Available: {'gini', 'entropy'}")

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)

    def _entropy(self, y):
        labels, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-9))

    def get_unique_values(self, X, feature):
        values = [features[feature] for features in X if feature in features]
        if is_list_of_lists(values):
            unique_values = np.unique([element for array in values for element in array])  # flatten
        else:
            unique_values = np.unique(values)
        return unique_values

    def _split(self, X, y, feature, thresh):
        values = [features[feature] for features in X if feature in features]
        if is_list_of_lists(values):
            left_idx = np.array([thresh in v for v in values])
        else:
            if isinstance(thresh, (int, float, np.number)):
                left_idx = values <= thresh
            else:
                left_idx = np.array([thresh == v for v in values])
        right_idx = ~left_idx

        X_left = [row for i, row in enumerate(X) if left_idx[i]]
        y_left = [label for i, label in enumerate(y) if left_idx[i]]
        X_right = [row for i, row in enumerate(X) if right_idx[i]]
        y_right = [label for i, label in enumerate(y) if right_idx[i]]
        return X_left, X_right, y_left, y_right
