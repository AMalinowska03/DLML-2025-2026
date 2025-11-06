import math

from decision_tree import DecisionTree
from collections import Counter
import numpy as np

class RandomForest:
    def __init__(self, tree_number=50):
        self.tree_number = tree_number
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for N in range(self.tree_number):
            tree = DecisionTree(features_number_to_compare=round(math.sqrt(len(X))))
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        final_predictions = []
        for i in range(len(X)):
            single_predictions = predictions[:, i] # predictions of each tree for single entry
            final_predictions.append(Counter(single_predictions).most_common(1)[0][0])
        return np.array(final_predictions)