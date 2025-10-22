feature_extractors = {
    'keywords': lambda m: [keyword['name'] for keyword in m['keywords']['keywords']],
    'genres': lambda m: [genre['name'] for genre in m['genres']],
    'production_companies': lambda m: [production_company['name'] for production_company in m['production_companies']],
    'original_language': lambda m: m['original_language'],
    'budget': lambda m: m['budget'],
    'overview': lambda m: m['overview'],
    'release_date': lambda m: m['release_date'],
    'vote_average': lambda m: m['vote_average'],
    'runtime': lambda m: m['runtime'],
    'popularity': lambda m: m['popularity'],
    'revenue': lambda m: m['revenue'],
    'actors': lambda m: [actor['name'] for actor in m['credits']['cast'] if actor['known_for_department'] == "Acting"]
}

class FeatureExtractor:
    def __init__(self, feature_extractors_to_include):
        self.feature_extractors_to_include = feature_extractors_to_include

    def extract(self, movies):
        movie_features = {}
        for movie_number, movie in movies.items():
            movie_features[movie_number] = {name: extractor(movie) for name, extractor in feature_extractors.items()
                                            if name in self.feature_extractors_to_include}
        return movie_features
