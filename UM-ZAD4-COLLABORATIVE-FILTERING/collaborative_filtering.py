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

        self.user_means = {self.users_dict[u]: np.mean(list(ratings.values())) for u, ratings in users_ratings.items()}
        self.normalized_ratings = {
            u: {m: (r - self.user_means[self.users_dict[u]]) for m, r in ratings.items()}
            for u, ratings in users_ratings.items()
        }

        self.flattened_ratings = [
            (self.users_dict[u], self.movies_dict[m], self.normalized_ratings[u][m])
            for u, ratings in self.users_ratings.items()
            for m, _ in ratings.items()
        ]

    def train(self, epoch=100):
        self.U = np.random.normal(scale=1. / self.num_factors, size=(self.num_users, self.num_factors))
        self.M = np.random.normal(scale=1. / self.num_factors, size=(self.num_movies, self.num_factors))

        for epoch_num in range(epoch):
            np.random.shuffle(self.flattened_ratings)
            for user_idx, movie_idx, rating in self.flattened_ratings:
                u_vec = self.U[user_idx]
                m_vec = self.M[movie_idx]

                prediction = self.predict_raw(user_idx, movie_idx)
                error = rating - prediction

                self.U[user_idx] += self.learning_rate * (error * m_vec - self.regularization_strength * u_vec)
                self.M[movie_idx] += self.learning_rate * (error * u_vec - self.regularization_strength * m_vec)

            logging.info(f"Epoch {epoch_num} loss: {self.compute_rmse()}")

    def predict_raw(self, user_idx, movie_idx):
        return np.dot(self.U[user_idx], self.M[movie_idx])

    def predict(self, user_id, movie_id):
        user_idx = self.users_dict[user_id]
        movie_idx = self.movies_dict[movie_id]
        pred = self.predict_raw(user_idx, movie_idx)
        pred += self.user_means[user_idx]
        return max(0, min(5, int(round(pred))))

    def compute_rmse(self):
        error = 0
        count = 0
        for user_id, movie_id, rating in self.flattened_ratings:
            error += (rating - self.predict_raw(user_id, movie_id)) ** 2
            count += 1
        return np.sqrt(error / count)
