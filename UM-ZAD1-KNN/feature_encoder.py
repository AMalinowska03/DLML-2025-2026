from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from encoders.enumerable_encoder import EnumerableEncoder


class FeatureEncoder:
    def __init__(self, movies_features):
        self.feature_encoders = {
            'keywords': lambda f: EnumerableEncoder(movies_features, 'keywords').encode(f),
            'genres': lambda f: EnumerableEncoder(movies_features, 'genres').encode(f),
            'production_companies': lambda f: OneHotEncoder(handle_unknown="ignore"),
            'overview': lambda f: TfidfVectorizer(max_features=300).fit_transform(f).toarray()
        }

    def encode(self, features):
        encoded_features = {}
        for movie_number, movie_features in features.items():
            encoded_features[movie_number] = {name: encoder(movie_features[name]) for name, encoder in self.feature_encoders.items()
                                              if name in movie_features}
        return encoded_features
