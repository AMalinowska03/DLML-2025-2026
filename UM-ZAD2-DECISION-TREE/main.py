import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from shared.movie_retriever import MovieRetriever
from shared.rating_retriever import RatingRetriever
from shared.feature_extractor import FeatureExtractor
from classifier_trainer import train_classifier
from shared.rating_prediction import RatingPrediction
from tree_drawer import draw_tree_png

features_to_extract = [
    'keywords',
    'genres',
    'production_companies',
    'original_language',
    'budget',
    'release_date',
    'vote_average',
    'runtime',
    'popularity',
    'revenue',
    'actors'
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] [Thread %(threadName)s] %(message)s",
)

def process_person(person_data):
    person, ratings, features = person_data
    accuracy, soft_accuracy, classifier = train_classifier(person, features, ratings)
    return person, accuracy, soft_accuracy, classifier


if __name__ == '__main__':
    movies = MovieRetriever().get_movies()
    ratings_per_person = RatingRetriever('../data/train.csv').get_ratings_per_person()

    features = FeatureExtractor(features_to_extract).extract(movies)

    classifier_per_person = {}
    accuracy_per_person = []
    soft_accuracy_per_person = []

    person_data_list = [(person, ratings, features) for person, ratings in ratings_per_person.items()]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_person, data) for data in person_data_list]

        for future in as_completed(futures):
            person, accuracy, soft_accuracy, classifier = future.result()
            classifier_per_person[person] = classifier
            accuracy_per_person.append(accuracy)
            soft_accuracy_per_person.append(soft_accuracy)

    logging.info(f"Accuracy per person (mean): {np.mean(accuracy_per_person):.2f}, soft_accuracy per person (mean): {np.mean(soft_accuracy_per_person):.2f}")

    draw_tree_png(classifier_per_person[481].root)

    RatingPrediction('../data/task.csv').submit_ratings_predictions(classifier_per_person, features)