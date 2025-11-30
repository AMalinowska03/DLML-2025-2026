import numpy as np
import logging


def calculate_pearson_similarity(user_a_id, user_b_id, user_ratings, min_common_movies=3):
    common_movies = []
    if user_a_id in user_ratings and user_b_id in user_ratings:
        movies_A = set(user_ratings[user_a_id].keys())
        movies_B = set(user_ratings[user_b_id].keys())
        common_movies = list(movies_A.intersection(movies_B))

    # logging.info(len(common_movies))
    if len(common_movies) < min_common_movies:
        return 0

    ratings_a = [user_ratings[user_a_id][movie] for movie in common_movies]
    ratings_b = [user_ratings[user_b_id][movie] for movie in common_movies]

    # korelacja pearsona:
    if len(np.unique(ratings_a)) == 1 or len(np.unique(ratings_b)) == 1:  # no correlation if all ratings are the same
        corr = 0
    else:
        corr = np.corrcoef(ratings_a, ratings_b)[0, 1]

    return corr


class CollaborativeFiltering:
    def __init__(self, users_ratings):
        self.user_similarities = {}
        self.average_ratings = {}

        self.users_ratings = users_ratings

    def calculate_similarities(self, min_common_movies=3):
        user_ids = list(self.users_ratings.keys())
        n_users = len(user_ids)

        for i in range(n_users):
            user_id_1 = user_ids[i]

            if user_id_1 not in self.user_similarities:
                self.user_similarities[user_id_1] = {}

            for j in range(i + 1, n_users):  # top triangle
                user_id_2 = user_ids[j]

                sim = calculate_pearson_similarity(user_id_1, user_id_2, self.users_ratings, min_common_movies)

                self.user_similarities[user_id_1][user_id_2] = sim

                if user_id_2 not in self.user_similarities:
                    self.user_similarities[user_id_2] = {}
                self.user_similarities[user_id_2][user_id_1] = sim

    def calculate_average_ratings(self):
        for user_id, ratings_dict in self.users_ratings.items():
            if ratings_dict:
                rates = list(ratings_dict.values())
                self.average_ratings[user_id] = np.mean(rates)

    def predict(self, user_id, movie_id, k=5):
        if len(self.user_similarities) == 0:
            self.calculate_similarities()
        if len(self.average_ratings) == 0:
            self.calculate_average_ratings()

        if user_id not in self.user_similarities:
            return None
        if movie_id in self.users_ratings[user_id]:
            return self.users_ratings[user_id][movie_id]

        neighbors = []  # users that watched that movie
        for other_user_id, movie_ratings in self.users_ratings.items():
            if other_user_id != user_id and movie_id in self.users_ratings[other_user_id]:
                similarity = self.user_similarities[user_id][other_user_id]

                if similarity > 0:
                    similarity **= 3
                    neighbor_avg_rating = self.average_ratings[other_user_id]
                    neighbor_rating = self.users_ratings[other_user_id][movie_id]
                    deviation = neighbor_rating - neighbor_avg_rating
                    neighbors.append((similarity, deviation))

        predicted_rating = self.average_ratings[user_id]

        if len(neighbors) == 0:
            return int(round(predicted_rating))  # return average rating if no similar users that watched this movie

        neighbors.sort(key=lambda x: x[0], reverse=True)
        top_k_neighbors = neighbors[:k]

        nominator = 0.0
        denominator = 0.0
        for similarity, deviation in top_k_neighbors:
            nominator += similarity * deviation
            denominator += abs(similarity)
        predicted_rating += nominator / denominator

        return max(0, min(5, int(round(predicted_rating))))
