from shared.rating_retriever import RatingRetriever
from collaborative_filtering import CollaborativeFiltering
from shared.rating_prediction import RatingPrediction
from cross_validation import cross_validation
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] [Thread %(threadName)s] %(message)s",
)

if __name__ == '__main__':
    ratings_per_person = RatingRetriever('../data/train.csv').get_ratings_per_person()

    train_accuracy, train_soft_accuracy, val_accuracy, val_soft_accuracy = cross_validation(ratings_per_person)
    logging.info(f"TRAIN - Accuracy per person (mean): {train_accuracy:.4f},  soft_accuracy per person (mean): {train_soft_accuracy:.4f}")
    logging.info(f"VALIDATE - Accuracy per person (mean): {val_accuracy:.4f},  soft_accuracy per person (mean): {val_soft_accuracy:.4f}")

    classifier = CollaborativeFiltering(users_ratings=ratings_per_person)
    classifier.train()

    RatingPrediction('../data/task.csv').submit_ratings_predictions_no_features(classifier)
