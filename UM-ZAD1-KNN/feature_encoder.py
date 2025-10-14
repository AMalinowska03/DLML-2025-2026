feature_encoders = {
    'keywords': lambda f: f,
    'genres': lambda f: f,
    'production_companies': lambda f: f
}

class FeatureEncoder:
    def encode(self, features):
        return {name: encoder(features[name]) for name, encoder in feature_encoders.items() if name in features}