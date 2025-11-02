import logging

from shared.cross_validation import cross_validation
from decision_tree import DecisionTree

cross_validation_folds = 5


def train_classifier(person, features, person_ratings):
    logging.info(f"Training classifier for person {person}...")

    X = [features[movie_number] for movie_number in person_ratings]
    y = [person_ratings[movie_number] for movie_number in person_ratings]

    classifier = DecisionTree()

    accuracy, soft_accuracy = cross_validation(X, y, cross_validation_folds, classifier)

    logging.info(f"Accuracy for person {person}: acc: {accuracy}, soft_acc: {soft_accuracy}")

    classifier.fit(X, y)

    return accuracy, soft_accuracy, classifier
