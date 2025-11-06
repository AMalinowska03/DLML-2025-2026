import math

from decision_tree import DecisionTree
from collections import Counter
import numpy as np

class RandomForest:
    def __init__(self, tree_number=1):
        self.tree_number = tree_number
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for N in range(self.tree_number):
            tree = DecisionTree(features_number_to_compare=round(math.sqrt(len(X))))
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        return np.array([self.predict_single(row) for row in X])

    def predict_single(self, row):
        predictions = np.array([tree.predict_single(row) for tree in self.trees])
        return Counter(predictions).most_common(1)[0][0]
