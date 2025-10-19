from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEncoder:
    def __init__(self, movies_features):
        self.movies_features = movies_features
        self.feature_encoders = {
            'keywords': lambda f: self.__vectorize(self.movies_features, f, 'keywords'),
            'genres': lambda f: self.__vectorize(self.movies_features, f, 'genres'),
            'production_companies': lambda f: self.__vectorize(self.movies_features, f, 'production_companies'),
            'original_language': lambda f: self.__vectorize(self.movies_features, f, 'original_language'),
            'overview': lambda f: self.__vectorize(self.movies_features, f, 'overview'),
            'budget': lambda f: [f],
            'release_date': lambda f: [datetime.strptime(f, '%Y-%m-%d').year],
            'vote_average': lambda f: [f],
            'runtime': lambda f: [f],
            'popularity': lambda f: [f],
            'revenue': lambda f: [f],
            'actors': lambda f: self.__vectorize(self.movies_features, f, 'actors')
        }

    def encode(self, features):
        encoded_features = {}
        for movie_number, movie_features in features.items():
            encoded_features[movie_number] = {name: encoder(movie_features[name]) for name, encoder in self.feature_encoders.items()
                                              if name in movie_features}
        return encoded_features

    def __vectorize(self, features, f, feature_name):
        vectorizer = TfidfVectorizer(max_features=100)
        vectorizer.fit([f[feature_name][0] for f in features.values()])
        return vectorizer.transform(f).toarray()[0]