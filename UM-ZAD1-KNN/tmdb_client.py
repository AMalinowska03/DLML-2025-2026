import tmdbsimple
import tmdbsimple as tmdb

class TMDBClient:
    def __init__(self, api_key: str):
        tmdb.API_KEY = api_key
        self.tmdb = tmdb

    def get_movie_by_id(self, movie_id: int):
        return tmdb.Movies(movie_id).info(append_to_response='keywords,credits')

    def get_genres(self):
        return tmdb.Genres().movie_list()['genres']
