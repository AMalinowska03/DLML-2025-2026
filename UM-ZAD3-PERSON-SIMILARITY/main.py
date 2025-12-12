from shared.rating_retriever import RatingRetriever
from collaborative_filtering import CollaborativeFiltering
from cross_validation import cross_validation
from shared.rating_prediction import RatingPrediction
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] [Thread %(threadName)s] %(message)s",
)

if __name__ == '__main__':
    ratings_per_person = RatingRetriever('../data/train.csv').get_ratings_per_person()

    predictor = CollaborativeFiltering(users_ratings=ratings_per_person)
    predictor.calculate_average_ratings()
    predictor.calculate_similarities(min_common_movies=20)

    cross_validation(ratings_per_person, min_common_movies=20)
    RatingPrediction('../data/task.csv').submit_ratings_predictions_no_features(predictor)
