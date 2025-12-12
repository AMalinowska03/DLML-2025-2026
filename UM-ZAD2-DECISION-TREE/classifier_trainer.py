import logging

from decision_tree import DecisionTree
from random_forest import RandomForest
from shared.cross_validation import cross_validation
from shared.stratified_split import train_test_split_stratified

cross_validation_folds = 6


def train_classifier(person, features, person_ratings, is_forest=False):
    classifier_type = "forest" if is_forest else "tree"
    logging.debug(f"Training classifier {classifier_type} for person {person}...")

    X = [features[movie_number] for movie_number in person_ratings]
    y = [person_ratings[movie_number] for movie_number in person_ratings]

    classifier = RandomForest() if is_forest else DecisionTree()

    train_accuracy, train_soft_accuracy, val_accuracy, val_soft_accuracy = cross_validation(X, y, cross_validation_folds, classifier)

    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, 0.2)

    classifier.fit(X_train, y_train)
    if not is_forest:
        classifier.prune(X_test, y_test)

    logging.debug(f"Classifier {classifier_type} for person {person} trained.")

    return train_accuracy, train_soft_accuracy, val_accuracy, val_soft_accuracy, classifier
