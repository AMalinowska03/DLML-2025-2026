import numpy as np

from feature_extractor import FeatureExtractor
from feature_encoder import FeatureEncoder
from movie_retriever import MovieRetriever
from rating_retriever import RatingRetriever
from knn import KnnClassifier
from rating_prediction import RatingPrediction

features_to_extract = ['keywords', 'genres']

if __name__ == '__main__':
    movies = MovieRetriever().get_movies()
    ratings_per_person = RatingRetriever('../data/train.csv').get_ratings_per_person()

    features = FeatureExtractor(features_to_extract).extract(movies)
    encoded_features = FeatureEncoder(features).encode(features)

    classifier_per_person = {}
    for person, ratings in ratings_per_person.items():
        X = []
        y = []
        for movie_number in ratings:
            X.append(np.concatenate(list(encoded_features[movie_number].values())))
            y.append(ratings[movie_number])
        classifier_per_person[person] = KnnClassifier(k=1, data=X, data_labels=y)

    RatingPrediction('../data/task.csv').submit_ratings_predictions(classifier_per_person, encoded_features)
