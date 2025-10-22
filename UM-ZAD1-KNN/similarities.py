import collections
import csv
import difflib
import itertools
import os
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tmdb_client import TMDBClient

similarities_csv = pd.read_csv("../data/genres.csv", sep=";", index_col='Name')


def string_similarity(string1, string2, prefix_weight=0.7, prefix_percent=0.2):
    if not string1 and not string2:
        return 1.0
    elif not string1 or not string2:
        return 0.0
    elif string1 == string2:
        return 1.0

    score = difflib.SequenceMatcher(None, string1, string2).ratio()
    if prefix_weight > 0:
        len1, len2 = len(string1), len(string2)
        longer_len = max(len1, len2)
        prefix_len = int(longer_len * prefix_percent)
        if prefix_len == 0:
            prefix_len = 1

        prefix1 = string1[:prefix_len]
        prefix2 = string2[:prefix_len]
        prefix_score = difflib.SequenceMatcher(None, prefix1, prefix2).ratio()
        return prefix_score * prefix_weight + score * (1.0 - prefix_weight)
    else:
        return score


def euclidean_distance(x1, x2):
    return np.sqrt((x1 - x2) ** 2)

def array_similarity(array1, array2, similarity_function=euclidean_distance, similarity_threshold=0.5):
    len1, len2 = len(array1), len(array2)
    if len1 == 0 and len2 == 0:
        return 1.0
    elif len1 == 0 or len2 == 1:
        return 0.0

    similarity_matrix = np.zeros(shape=(len1, len2))
    for i in range(len1):
        for j in range(len2):
            similarity_matrix[i, j] = similarity_function(array1[i], array2[j])

    best_matches_for_1 = np.max(similarity_matrix, axis=1)
    avg_sim_1_to_2 = np.mean(best_matches_for_1)

    best_matches_for_2 = np.max(similarity_matrix, axis=0)
    avg_sim_2_to_1 = np.mean(best_matches_for_2)

    return (avg_sim_1_to_2 + avg_sim_2_to_1) / 2.0

def genres_similarity(genre1, genre2):
    return similarities_csv[genre1][genre2]

def generate_genres_similarities():
    client = TMDBClient(api_key=os.getenv("TMDB_API_KEY"))
    genres_list = client.get_genres()

    discover = client.tmdb.Discover()
    occurances_count = collections.Counter()
    co_occurances_count = collections.defaultdict(collections.Counter)
    TOTAL_PAGE_COUNT = 500
    for page in range(1, TOTAL_PAGE_COUNT + 1):
        if page % 50 == 0:
            print(f"  Przetwarzanie strony {page}/{TOTAL_PAGE_COUNT}...")
        movies = discover.movie(
            page=page,
            sort_by='popularity.desc',
            **{'vote_count.gte': 100}
        )
        for movie in movies['results']:
            movie_genre_ids = movie.get('genre_ids', [])
            for genre_id in movie_genre_ids:
                occurances_count[genre_id] += 1.0
            for genre1, genre2 in itertools.combinations(movie_genre_ids, 2):
                co_occurances_count[genre1][genre2] += 1.0
                co_occurances_count[genre2][genre1] += 1.0

    similarity_scores = collections.defaultdict(dict)

    try:
        with open("../data/genres.csv", 'w', newline='', encoding='utf-8') as csvfile:
            # Określenie nagłówka kolumny
            fieldnames = ['Name']
            for genre in genres_list:
                fieldnames.append(genre['name'])

            # Używamy DictWriter dla łatwiejszego zapisu z nagłówkiem
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")

            # Zapisz nagłówek
            writer.writeheader()


            # Zapisz każdy gatunek w nowym wierszu
            for genre in genres_list:
                row = {'Name': genre['name']}

                for genre_col in genres_list:
                    if genre_col['name'] == genre['name']:
                        row[genre_col['name']] = 1.0
                    else:
                        co_occurance_count = co_occurances_count[genre['id']][genre_col['id']]
                        if co_occurance_count == 0:
                            row[genre_col['name']] = 0.0
                        elif genre['name'] in similarity_scores and genre_col['name'] in similarity_scores[genre['name']]:
                            row[genre_col['name']] = similarity_scores[genre['name']][genre_col['name']]
                        else:
                            union = occurances_count[genre['id']] + occurances_count[genre_col['id']] - co_occurance_count
                            score = co_occurance_count / union if union > 0 else 0.0
                            score **= 0.15

                            similarity_scores[genre['name']][genre_col['name']] = round(score, 2)
                            similarity_scores[genre_col['name']][genre['name']] = round(score, 2)
                            row[genre_col['name']] = round(score, 2)

                writer.writerow(row)

    except IOError as e:
        print(f"BŁĄD: Nie udało się zapisać pliku CSV. Powód: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas zapisu: {e}", file=sys.stderr)
        print("-----------------------------------")
        traceback.print_exc(file=sys.stdout)


def jaccard_similarity(list1, list2):
    """Oblicza podobieństwo Jaccarda dla dwóch list."""
    set1 = set(list1)
    set2 = set(list2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 1.0

    return intersection / union

def numeric_similarity(norm_num1, norm_num2):
    """Oblicza podobieństwo dla dwóch liczb znormalizowanych do [0, 1]."""
    return 1.0 - abs(norm_num1 - norm_num2)

def process_similarity(feature_key, value1, value2):
    match feature_key:
        case 'genres':
            return array_similarity(value1, value2, similarity_function=genres_similarity)
        case 'keywords' | 'production_companies' | 'actors':
            return jaccard_similarity(value1, value2)
        case 'overview':
            return string_similarity(value1, value2)
        case 'original_language':
            return string_similarity(value1, value2, prefix_weight=0)
        case 'budget' | 'release_date' | 'vote_average' | 'runtime' | 'popularity' | 'revenue':
            return numeric_similarity(value1, value2)


feature_weights = {
    'keywords': 0.15,
    'genres': 0.3,
    'production_companies': 0.05,
    'original_language': 0.1,
    'overview': 0.0,
    'budget': 0.05,
    'release_date': 0.1,
    'vote_average': 0.15,
    'runtime': 0.02,
    'popularity': 0.05,
    'revenue': 0.03,
    'actors': 0.05
}

def process_similarity_full(feature_vector1, feature_vector2):
    total_similarity = 0.0
    count = 0
    for key in feature_weights.keys():
        if feature_weights[key] == 0:
            score = 0
        else:
            count += 1
            score = process_similarity(key, feature_vector1[key], feature_vector2[key])
        total_similarity += score*feature_weights[key]
    return total_similarity