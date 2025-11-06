from datetime import datetime

feature_extractors = {
    'keywords': lambda m: [keyword['name'] for keyword in m['keywords']['keywords']],
    'genres': lambda m: [genre['name'] for genre in m['genres']],
    'production_companies': lambda m: [production_company['name'] for production_company in m['production_companies']],
    'original_language': lambda m: m['original_language'],
    'budget': lambda m: m['budget'],
    'overview': lambda m: m['overview'],
    'release_date': lambda m: datetime.strptime(m['release_date'], '%Y-%m-%d').year,
    'vote_average': lambda m: m['vote_average'],
    'runtime': lambda m: m['runtime'],
    'popularity': lambda m: m['popularity'],
    'revenue': lambda m: m['revenue'],
    'actors': lambda m: [cast['name'] for cast in m['credits']['cast'] if cast['known_for_department'] == "Acting" and cast['popularity'] > 1],
    'spoken_languages': lambda m: [language['iso_639_1'] for language in m['spoken_languages']],
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
