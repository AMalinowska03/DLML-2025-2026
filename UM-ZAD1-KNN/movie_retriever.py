import json
import os
from dotenv import load_dotenv
import pandas as pd

from tmdb_client import TMDBClient

load_dotenv()

movies_file_path = '../data/movie.csv'
cache_file_path = '../data/movies_cache.json'

class MovieRetriever:
    def __init__(self):
        self.tmdb_client = TMDBClient(api_key=os.getenv("TMDB_API_KEY"))
        self.movies_df = pd.read_csv(movies_file_path, delimiter=';', header=None)

    def get_movies(self):
        if os.path.exists(cache_file_path):
            print("Loading movies from cache file...")
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                json_movies = json.load(f)
                int_keyed_dict = {}
                for i in json_movies.keys():
                    int_keyed_dict[int(i)] = json_movies[i]
                return int_keyed_dict

        print("Retrieving movies from API...")
        movies = {}
        for movie in self.movies_df.itertuples(index=False):
            movie_local_id = movie[0]
            movie_tmdb_id = movie[1]
            movie_title = movie[2]

            movies[movie_local_id] = self.tmdb_client.get_movie_by_id(movie_tmdb_id)
            print(f"#{movie_local_id} Movie \"{movie_title}\" (TMDB ID: {movie_tmdb_id}) retrieved.")

        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(movies, f, ensure_ascii=False, indent=2)

        print("Movies saved to cache.")
        return movies
