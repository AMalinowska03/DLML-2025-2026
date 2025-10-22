from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


class FeatureEncoder:
    def __init__(self, movies_features):
        self.movies_features = movies_features
        self.scalers = {}

        self.numeric_features = [
            'budget', 'release_date', 'vote_average',
            'runtime', 'popularity', 'revenue'
        ]
        temp_numeric_data = {key: [] for key in self.numeric_features}

        for movie_id, features in movies_features.items():
            for key, value in features.items():
                if key == 'release_date':
                    temp_numeric_data[key].append(datetime.strptime(value, '%Y-%m-%d').year if value else 0)
                elif key in self.numeric_features:
                    temp_numeric_data[key].append(value)

        for key in self.numeric_features:
            if not temp_numeric_data[key]:
                continue
            scaler = MinMaxScaler()
            scaler.fit(np.array(temp_numeric_data[key]).reshape(-1, 1))
            self.scalers[key] = scaler

        self.feature_encoders = {
            'keywords': lambda f: f,
            'genres': lambda f: f,
            'production_companies': lambda f: f,
            'original_language': lambda f: f,
            'overview': lambda f: f,
            'budget': lambda f: self.scalers['budget'].transform([[f]])[0][0],
            'release_date': lambda f: self.scalers['release_date'].transform([[datetime.strptime(f, '%Y-%m-%d').year if f else 0]])[0][0],
            'vote_average': lambda f: self.scalers['vote_average'].transform([[f]])[0][0],
            'runtime': lambda f: self.scalers['runtime'].transform([[f]])[0][0],
            'popularity': lambda f: self.scalers['popularity'].transform([[f]])[0][0],
            'revenue': lambda f: self.scalers['revenue'].transform([[f]])[0][0],
            'actors': lambda f: f
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