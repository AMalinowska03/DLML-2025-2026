from decision_tree import DecisionTree
from collections import Counter
import numpy as np

from shared.stratified_split import train_test_split_stratified

class RandomForest:
    def __init__(self, tree_number=25):
        self.is_forest = True
        self.tree_number = tree_number
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for N in range(self.tree_number):
            tree = DecisionTree(features_number_to_compare=3)
            X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, 0.2)
            tree.fit(X_train, y_train)
            tree.prune(X_test, y_test)
            self.trees.append(tree)

    def predict(self, X):
        return np.array([self.predict_single(row) for row in X])

    def predict_single(self, row):
        predictions = np.array([tree.predict_single(row) for tree in self.trees])
        return Counter(predictions).most_common(1)[0][0]
