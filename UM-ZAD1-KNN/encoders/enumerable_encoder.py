import numpy as np

class EnumerableEncoder():
    def __init__(self, movies_features, feature):
        self.movies_features = movies_features
        self.feature = feature
        all_elements = sorted({g for f in movies_features.values() for g in f[feature]})
        self.element_to_idx = {g: i for i, g in enumerate(all_elements)}

    def encode(self, feature):
        vec = np.zeros(len(self.element_to_idx))
        for e in feature:
            vec[self.element_to_idx[e]] = 1
        return vec
