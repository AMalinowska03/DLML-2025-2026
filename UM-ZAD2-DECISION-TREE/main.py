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
    'original_language',
    'origin_country',
    'budget',
    'release_date',
    'vote_average',
    'runtime',
    'popularity',
    'revenue',
    'actors',
    'super_stars',
    'main_character_gender',
    'directors',
    'spoken_languages',
    'production_companies'
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] [Thread %(threadName)s] %(message)s",
)


def process_person(person_data, is_forest=False):
    person, ratings, features = person_data
    accuracy, soft_accuracy, classifier = train_classifier(person, features, ratings, is_forest)
    return person, accuracy, soft_accuracy, classifier


def experiment(is_forest=False):
    classifier_type = "forest" if is_forest else "tree"

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_person, data, is_forest) for data in person_data_list]

        for idx, future in enumerate(as_completed(futures)):
            person, accuracy, soft_accuracy, classifier = future.result()
            classifier_per_person[person] = classifier
            accuracy_per_person.append(accuracy)
            soft_accuracy_per_person.append(soft_accuracy)
            logging.info(f"Processed - {idx + 1}/{len(person_data_list)} | Person: {person}, accuracy: {accuracy}, soft_accuracy: {soft_accuracy} - {classifier_type}")

    logging.info(f"Accuracy per person (mean): {np.mean(accuracy_per_person):.2f},  soft_accuracy per person (mean): {np.mean(soft_accuracy_per_person):.2f}")

    if not is_forest:
        draw_tree_png(classifier_per_person[169].root, filename=f"tree.png")

    filename = f"submission_{classifier_type}"
    RatingPrediction('../data/task.csv').submit_ratings_predictions(classifier_per_person, features, filename)


if __name__ == '__main__':
    movies = MovieRetriever().get_movies()
    ratings_per_person = RatingRetriever('../data/train.csv').get_ratings_per_person()

    features = FeatureExtractor(features_to_extract).extract(movies)

    classifier_per_person = {}
    accuracy_per_person = []
    soft_accuracy_per_person = []

    person_data_list = [(person, ratings, features) for person, ratings in ratings_per_person.items()]

    experiment(False)  # tree
    # experiment(True)  # forest

