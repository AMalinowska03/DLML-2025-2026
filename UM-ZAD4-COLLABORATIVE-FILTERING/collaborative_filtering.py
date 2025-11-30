import logging

import numpy as np


class CollaborativeFiltering:
    def __init__(self, users_ratings, num_factors=10, learning_rate=0.01, regularization_strength=0.01):
        self.U = None
        self.M = None
        self.users_ratings = users_ratings
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.num_users = len(users_ratings)
        self.users_dict = {user_id: index for index, user_id in enumerate(users_ratings.keys())}

        unique_movies = np.unique([movie_id for ratings in users_ratings.values() for movie_id in ratings.keys()])
        self.num_movies = len(unique_movies)
        self.movies_dict = {movie_id: index for index, movie_id in enumerate(unique_movies)}

    def train(self, epoch=100):
        self.U = np.random.rand(self.num_users, self.num_factors)
        self.M = np.random.rand(self.num_movies, self.num_factors)

        for epoch_num in range(epoch):
            logging.info(f"Epoch {epoch_num} training started...")
            for user_id, ratings_dict in self.users_ratings.items():
                for movie_id, rating in ratings_dict.items():
                    u = self.users_dict[user_id]
                    m = self.movies_dict[movie_id]

                    prediction = self.predict_raw(user_id, movie_id)
                    error = rating - prediction

                    u_vec = self.U[u]
                    m_vec = self.M[m]

                    self.U[u] += self.learning_rate * (error * m_vec - self.regularization_strength * u_vec)
                    self.M[m] += self.learning_rate * (error * u_vec - self.regularization_strength * m_vec)

    def predict_raw(self, user_id, movie_id):
        u = self.users_dict[user_id]
        m = self.movies_dict[movie_id]
        return np.dot(self.U[u], self.M[m])

    def predict(self, user_id, movie_id):
        pred = self.predict_raw(user_id, movie_id)
        return max(0, min(5, int(round(pred))))
