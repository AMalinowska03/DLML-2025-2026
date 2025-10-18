import pandas as pd

class RatingRetriever:
    def __init__(self, ratings_file_path):
        self.ratings = pd.read_csv(ratings_file_path, delimiter=';', header=None)

    def get_ratings_per_person(self):
        ratings_per_person = {}
        for rating in self.ratings.itertuples(index=False):
            if rating[1] not in ratings_per_person:
                ratings_per_person[rating[1]] = {}
            ratings_per_person[rating[1]][rating[2]] = rating[3]
        return ratings_per_person
