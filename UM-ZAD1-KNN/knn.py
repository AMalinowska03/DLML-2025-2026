import numpy as np

class KnnClassifier():
    def __init__(self, k, data, data_labels):
        self.k = k
        self.data = data
        self.data_labels = data_labels

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        distances = []
        for i in range(len(self.data)):
            distance = self._euclidean_distance(x, self.data[i])
            label = self.data_labels[i]
            distances.append((distance, label))
        distances.sort(key = lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:self.k]]
        return int(max(set(k_nearest_labels), key=k_nearest_labels.count))

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((np.array(x1) - np.array(x2))**2))
