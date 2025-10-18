import os
from dotenv import load_dotenv
import pandas as pd

from tmdb_client import TMDBClient

load_dotenv()

movies_file_path = '../data/movie.csv'

class MovieRetriever:
    def __init__(self):
        self.tmdb_client = TMDBClient(api_key=os.getenv("TMDB_API_KEY"))
        self.movies = pd.read_csv(movies_file_path, delimiter=';', header=None)

    def get_movies(self):
        movies = {}
        for movie in self.movies.itertuples(index=False):
            movies[movie[0]] = self.tmdb_client.get_movie_by_id(movie[1])
            print(f"#{movie[0]} Movie \"{movie[2]}\" with id {movie[1]} retrieved.")
        return movies
