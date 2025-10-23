import numpy as np
import pandas as pd

class RatingPrediction:
    def __init__(self, ratings_file_path):
        self.ratings = pd.read_csv(ratings_file_path, delimiter=';', header=None)

    def submit_ratings_predictions(self, classifiers_per_person, encoded_features, best_features_per_person):
        self.ratings[3] = self.ratings[3].astype(object)
        for idx, rating in self.ratings.iterrows():
            person = rating[1]
            movie = rating[2]
            filtered_features = {
                outer_k: {k: v for k, v in inner.items() if k in best_features_per_person[person]}
                for outer_k, inner in encoded_features.items()
            }
            self.ratings.at[idx, 3] = str(classifiers_per_person[person].predict_single(filtered_features[movie]))
        self.ratings.to_csv("./submission.csv", sep=';', index=False, header=False)