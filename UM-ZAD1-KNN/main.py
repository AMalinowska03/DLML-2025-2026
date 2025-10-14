import os
from dotenv import load_dotenv

from tmdb_client import TMDBClient
from feature_extractor import FeatureExtractor
from feature_encoder import FeatureEncoder

load_dotenv()
api_key = os.getenv("TMDB_API_KEY")

features_to_extract = ['genres']

if __name__ == '__main__':
    tmdb_client = TMDBClient(api_key=api_key)
    feature_extractor = FeatureExtractor(features_to_extract)
    feature_encoder = FeatureEncoder()

    movie = tmdb_client.get_movie_by_id(664413)
    features = feature_extractor.extract(movie)

    print(features)

    encoded_features = feature_encoder.encode(features)
    print(encoded_features)




