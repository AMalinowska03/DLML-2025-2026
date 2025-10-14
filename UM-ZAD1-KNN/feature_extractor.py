feature_extractors = {
    'keywords': lambda m: [keyword['name'] for keyword in m['keywords']['keywords']],
    'genres': lambda m: [genre['name'] for genre in m['genres']],
    'production_companies': lambda m: [production_company['name'] for production_company in m['production_companies']]
}

class FeatureExtractor:
    def __init__(self, feature_extractors_to_include):
        self.feature_extractors_to_include = feature_extractors_to_include

    def extract(self, movie):
        return {name: extractor(movie) for name, extractor in feature_extractors.items() if name in self.feature_extractors_to_include}