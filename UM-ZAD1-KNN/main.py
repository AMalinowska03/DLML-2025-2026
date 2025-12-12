import logging

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from feature_extractor import FeatureExtractor
from feature_encoder import FeatureEncoder
from movie_retriever import MovieRetriever
from rating_retriever import RatingRetriever
from rating_prediction import RatingPrediction
from similarities import generate_genres_similarities
from feature_selection import select_best_features, load_best_features, save_best_features
from classifier_trainer import train_classifier

features_to_extract = [
    'keywords', 'genres', 'production_companies', 'original_language',
    'overview', 'budget', 'release_date', 'vote_average',
    'runtime', 'popularity', 'revenue', 'actors'
]
k_neighbors = 15
cross_validation_folds = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] [Thread %(threadName)s] %(message)s",
)

def process_person(person_data):
    person, ratings, encoded_features, best_features_per_person = person_data

    if person not in best_features_per_person:
        logging.info(f"Selecting best features for person {person}...")
        best_features, score = select_best_features(person, encoded_features, ratings)
        logging.info(f"Selected features for person {person}: {best_features} with score: {score}")
    else:
        best_features = best_features_per_person[person]

    filtered_features = {
        outer_k: {k: v for k, v in inner.items() if k in best_features}
        for outer_k, inner in encoded_features.items()
    }

    train_accuracy, train_soft_accuracy, val_accuracy, val_soft_accuracy, classifier = train_classifier(person, encoded_features, ratings, find_best_k=True)

    return person, best_features, train_accuracy, train_soft_accuracy, val_accuracy, val_soft_accuracy, classifier


if __name__ == '__main__':
    # generate_genres_similarities()
    movies = MovieRetriever().get_movies()
    ratings_per_person = RatingRetriever('../data/train.csv').get_ratings_per_person()

    features = FeatureExtractor(features_to_extract).extract(movies)
    encoded_features = FeatureEncoder(features).encode(features)

    best_features_per_person = load_best_features()
    classifier_per_person = {}
    train_accuracy_per_person = []
    train_soft_accuracy_per_person = []
    val_accuracy_per_person = []
    val_soft_accuracy_per_person = []

    person_data_list = [(person, ratings, encoded_features, best_features_per_person) for person, ratings in ratings_per_person.items()]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_person, data) for data in person_data_list]

        for future in as_completed(futures):
            person, best_features, train_accuracy, train_soft_accuracy, val_accuracy, val_soft_accuracy, classifier = future.result()
            best_features_per_person[person] = best_features
            classifier_per_person[person] = classifier
            train_accuracy_per_person.append(train_accuracy)
            train_soft_accuracy_per_person.append(train_soft_accuracy)
            val_accuracy_per_person.append(val_accuracy)
            val_soft_accuracy_per_person.append(val_soft_accuracy)

    save_best_features(best_features_per_person)

    logging.info(f"TRAIN - Accuracy per person (mean): {np.mean(train_accuracy_per_person):.4f}, soft_accuracy per person (mean): {np.mean(train_soft_accuracy_per_person):.4f}")
    logging.info(f"VALIDATE - Accuracy per person (mean): {np.mean(val_accuracy_per_person):.4f}, soft_accuracy per person (mean): {np.mean(val_soft_accuracy_per_person):.4f}")

    RatingPrediction('../data/task.csv').submit_ratings_predictions(classifier_per_person, encoded_features, best_features_per_person)
