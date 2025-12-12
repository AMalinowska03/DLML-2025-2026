import json
import logging
import os

from classifier_trainer import train_classifier

FEATURES_FILE = "../data/best_features_per_person.json"

cross_validation_folds = 10

def select_best_features(person, features, ratings_per_person):
    all_features = set(next(iter(features.values())).keys())
    selected = set([])
    best_score = 0

    while True:
        improved = False
        best_feature = None
        best_feature_score = best_score

        for feature in (all_features - selected):
            candidate_features = list(selected) + list([feature])
            filtered_features = {
                outer_k: {k: v for k, v in inner.items() if k in candidate_features}
                for outer_k, inner in features.items()
            }
            logging.info(f"Training classifier for person {person} with features: {candidate_features}...")
            _, _, score, _, _ = train_classifier(person, filtered_features, ratings_per_person)
            if score > best_feature_score:
                improved = True
                best_feature_score = score
                best_feature = feature

        if improved:
            selected.add(best_feature)
            best_score = best_feature_score
        else:
            break

    return selected, best_score


def save_best_features(best_features_dict, path=FEATURES_FILE):
    if not os.path.exists(path):
        serializable_dict = {person: list(features) for person, features in best_features_dict.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable_dict, f, indent=4)
        logging.info(f"Saved best features per person to {path}")


def load_best_features(path=FEATURES_FILE):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logging.info(f"Loaded best features per person from {path}")
        return {int(person): set(features) for person, features in data.items()}
    else:
        return {}
