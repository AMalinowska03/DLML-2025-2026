import numpy as np
from sklearn.preprocessing import MaxAbsScaler

from feature_extractor import FeatureExtractor
from feature_encoder import FeatureEncoder
from movie_retriever import MovieRetriever
from rating_retriever import RatingRetriever
from knn import KnnClassifier
from rating_prediction import RatingPrediction
from cross_validation import cross_validation
from similarities import generate_genres_similarities

features_to_extract = [
    'keywords',
    'genres',
    'production_companies',
    'original_language',
    'overview',
    'budget',
    'release_date',
    'vote_average',
    'runtime',
    'popularity',
    'revenue',
    'actors'
]
k_neighbors = 15
cross_validation_folds = 10

if __name__ == '__main__':
    # generate_genres_similarities()
    movies = MovieRetriever().get_movies()
    ratings_per_person = RatingRetriever('../data/train.csv').get_ratings_per_person()

    features = FeatureExtractor(features_to_extract).extract(movies)
    encoded_features = FeatureEncoder(features).encode(features)

    classifier_per_person = {}
    accuracy_per_person = []
    for person, ratings in ratings_per_person.items():
        X = []
        y = []
        for movie_number in ratings:
            # X.append(np.concatenate(list(encoded_features[movie_number].values())))
            X.append(encoded_features[movie_number])
            y.append(ratings[movie_number])
        print(f"Calculating accuracy for person {person}...")
        # X_scaled = MaxAbsScaler().fit_transform(X)
        accuracy_per_person.append(cross_validation(X, y, cross_validation_folds, k_neighbors))
        classifier_per_person[person] = KnnClassifier(k=k_neighbors, data=X, data_labels=y)

    print(f"Accuracy per person (mean): {np.mean(accuracy_per_person)}")

    RatingPrediction('../data/task.csv').submit_ratings_predictions(classifier_per_person, encoded_features)
