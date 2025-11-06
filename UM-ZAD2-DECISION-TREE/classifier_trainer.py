import logging

from shared.cross_validation import cross_validation
from decision_tree import DecisionTree
from random_forest import RandomForest

cross_validation_folds = 6


def train_classifier(person, features, person_ratings, is_forest=False):
    classifier_type = "forest" if is_forest else "tree"
    logging.info(f"Training classifier {classifier_type} for person {person}...")

    X = [features[movie_number] for movie_number in person_ratings]
    y = [person_ratings[movie_number] for movie_number in person_ratings]

    classifier = RandomForest() if is_forest else DecisionTree()

    accuracy, soft_accuracy = cross_validation(X, y, cross_validation_folds, classifier, process_async=False)

    logging.info(f"Accuracy for person {person}: acc: {accuracy}, soft_acc: {soft_accuracy} - {classifier_type}")

    classifier.fit(X, y)

    return accuracy, soft_accuracy, classifier
