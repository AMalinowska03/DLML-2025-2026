import numpy as np
from similarities import process_similarity_full

class KnnClassifier():
    def __init__(self, k):
        self.k = k
        self.data = None
        self.data_labels = None

    def fit(self, data, data_labels):
        self.data = np.array(data)
        self.data_labels = np.array(data_labels)

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        distances = []
        for i in range(len(self.data)):
            distance = process_similarity_full(x, self.data[i])
            label = self.data_labels[i]
            distances.append((distance, label))
        distances.sort(key = lambda x: x[0], reverse=True)
        k_nearest_labels = [label for _, label in distances[:self.k]]
        return int(max(set(k_nearest_labels), key=k_nearest_labels.count))

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((np.array(x1) - np.array(x2))**2))
