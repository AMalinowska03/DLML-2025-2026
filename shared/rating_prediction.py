import pandas as pd

class RatingPrediction:
    def __init__(self, ratings_file_path):
        self.ratings = pd.read_csv(ratings_file_path, delimiter=';', header=None)

    def submit_ratings_predictions(self, classifiers_per_person, features):
        self.ratings[3] = self.ratings[3].astype(object)
        for idx, rating in self.ratings.iterrows():
            person = rating[1]
            movie = rating[2]
            self.ratings.at[idx, 3] = str(classifiers_per_person[person].predict_single(features[movie]))
        self.ratings.to_csv("./submission.csv", sep=';', index=False, header=False)