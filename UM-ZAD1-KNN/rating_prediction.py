import numpy as np
import pandas as pd

class RatingPrediction:
    def __init__(self, ratings_file_path):
        self.ratings = pd.read_csv(ratings_file_path, delimiter=';', header=None)

    def submit_ratings_predictions(self, classifiers_per_person, encoded_features):
        self.ratings[3] = self.ratings[3].astype(object)
        for idx, rating in self.ratings.iterrows():
            self.ratings.at[idx, 3] = str(classifiers_per_person[rating[1]].predict(np.concatenate(list(encoded_features[rating[2]].values()))))
        self.ratings.to_csv("../data/submission.csv", sep=';', index=False, header=False)