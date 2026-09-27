"""Feature engineering for entity pairs."""

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, Jaro
from typing import Dict, Any
try:
    from .normalization import (
        normalize_name, normalize_name_aggressive, normalize_address,
        normalize_country, get_tokens, get_char_ngrams, extract_numbers,
        normalize_transliterate
    )
except ImportError:
    from normalization import (
        normalize_name, normalize_name_aggressive, normalize_address,
        normalize_country, get_tokens, get_char_ngrams, extract_numbers,
        normalize_transliterate
    )


def jaccard_similarity(set1, set2):
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def cosine_similarity_tokens(tokens1, tokens2):
    """Simple cosine similarity based on token overlap."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    all_tokens = tokens1 | tokens2
    if not all_tokens:
        return 0.0

    vec1 = np.array([1 if t in tokens1 else 0 for t in all_tokens])
    vec2 = np.array([1 if t in tokens2 else 0 for t in all_tokens])

    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def compute_name_features(name1: str, name2: str) -> Dict[str, float]:
    """Compute name similarity features."""
    features = {}

    # Normalized versions
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)

    norm_agg1 = normalize_name_aggressive(name1)
    norm_agg2 = normalize_name_aggressive(name2)

    # Transliterated versions
    trans1 = normalize_transliterate(name1)
    trans2 = normalize_transliterate(name2)

    # Exact match
    features['name_exact'] = 1.0 if norm1 == norm2 else 0.0
    features['name_exact_aggressive'] = 1.0 if norm_agg1 == norm_agg2 else 0.0

    # String similarities
    features['name_levenshtein'] = Levenshtein.normalized_similarity(norm1, norm2) if norm1 and norm2 else 0.0
    features['name_jaro'] = Jaro.normalized_similarity(norm1, norm2) if norm1 and norm2 else 0.0
    features['name_ratio'] = fuzz.ratio(norm1, norm2) / 100.0 if norm1 and norm2 else 0.0
    features['name_partial_ratio'] = fuzz.partial_ratio(norm1, norm2) / 100.0 if norm1 and norm2 else 0.0

    # Token-based
    tokens1 = get_tokens(norm1)
    tokens2 = get_tokens(norm2)
    features['name_token_jaccard'] = jaccard_similarity(tokens1, tokens2)
    features['name_token_cosine'] = cosine_similarity_tokens(tokens1, tokens2)
    features['name_token_overlap_count'] = len(tokens1 & tokens2)
    features['name_token_sort_ratio'] = fuzz.token_sort_ratio(norm1, norm2) / 100.0 if norm1 and norm2 else 0.0

    # Character n-grams
    ngrams1 = get_char_ngrams(norm1, 3)
    ngrams2 = get_char_ngrams(norm2, 3)
    features['name_char3gram_jaccard'] = jaccard_similarity(ngrams1, ngrams2)

    # Transliterated similarity (helps with cross-script)
    features['name_trans_levenshtein'] = Levenshtein.normalized_similarity(trans1, trans2) if trans1 and trans2 else 0.0

    # Length features
    len1 = len(norm1)
    len2 = len(norm2)
    features['name_len_diff'] = abs(len1 - len2)
    features['name_len_ratio'] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

    return features


def compute_address_features(addr1: str, addr2: str) -> Dict[str, float]:
    """Compute address similarity features."""
    features = {}

    # Normalized versions
    norm1 = normalize_address(addr1)
    norm2 = normalize_address(addr2)

    # Handle empty addresses
    empty1 = (not norm1)
    empty2 = (not norm2)
    features['address_both_empty'] = 1.0 if (empty1 and empty2) else 0.0
    features['address_one_empty'] = 1.0 if (empty1 != empty2) else 0.0

    if empty1 or empty2:
        # If one is empty, set other features to neutral/low values
        features['address_exact'] = 0.0
        features['address_levenshtein'] = 0.0
        features['address_jaro'] = 0.0
        features['address_ratio'] = 0.0
        features['address_token_jaccard'] = 0.0
        features['address_token_cosine'] = 0.0
        features['address_char3gram_jaccard'] = 0.0
        features['address_number_overlap'] = 0.0
        features['address_number_match_ratio'] = 0.0
        return features

    # Exact match
    features['address_exact'] = 1.0 if norm1 == norm2 else 0.0

    # String similarities
    features['address_levenshtein'] = Levenshtein.normalized_similarity(norm1, norm2)
    features['address_jaro'] = Jaro.normalized_similarity(norm1, norm2)
    features['address_ratio'] = fuzz.ratio(norm1, norm2) / 100.0

    # Token-based
    tokens1 = get_tokens(norm1)
    tokens2 = get_tokens(norm2)
    features['address_token_jaccard'] = jaccard_similarity(tokens1, tokens2)
    features['address_token_cosine'] = cosine_similarity_tokens(tokens1, tokens2)

    # Character n-grams
    ngrams1 = get_char_ngrams(norm1, 3)
    ngrams2 = get_char_ngrams(norm2, 3)
    features['address_char3gram_jaccard'] = jaccard_similarity(ngrams1, ngrams2)

    # Number overlap (house numbers, PIN codes, etc.)
    numbers1 = extract_numbers(norm1)
    numbers2 = extract_numbers(norm2)
    features['address_number_overlap'] = len(numbers1 & numbers2)

    if numbers1 or numbers2:
        features['address_number_match_ratio'] = jaccard_similarity(numbers1, numbers2)
    else:
        features['address_number_match_ratio'] = 0.0

    return features


def compute_country_features(country1: str, country2: str) -> Dict[str, float]:
    """Compute country features."""
    features = {}

    norm1 = normalize_country(country1)
    norm2 = normalize_country(country2)

    features['country_exact_match'] = 1.0 if norm1 == norm2 else 0.0
    features['country_mismatch'] = 1.0 if (norm1 and norm2 and norm1 != norm2) else 0.0

    return features


def compute_cross_features(name_features: Dict[str, float],
                          address_features: Dict[str, float],
                          country_features: Dict[str, float]) -> Dict[str, float]:
    """Compute cross-field combined features."""
    features = {}

    # Strong signals: both name and address match well
    features['name_addr_both_high'] = float(
        name_features['name_levenshtein'] > 0.85 and
        address_features['address_levenshtein'] > 0.85
    )

    # Name exact but address fuzzy
    features['name_exact_addr_fuzzy'] = float(
        name_features['name_exact'] == 1.0 and
        address_features['address_levenshtein'] > 0.5 and
        address_features['address_levenshtein'] < 0.95
    )

    # Address exact but name fuzzy
    features['addr_exact_name_fuzzy'] = float(
        address_features['address_exact'] == 1.0 and
        name_features['name_levenshtein'] > 0.5 and
        name_features['name_levenshtein'] < 0.95
    )

    # Weighted average
    features['weighted_similarity'] = (
        0.6 * name_features['name_levenshtein'] +
        0.3 * address_features['address_levenshtein'] +
        0.1 * country_features['country_exact_match']
    )

    # Token-based combined
    features['token_combined'] = (
        0.6 * name_features['name_token_jaccard'] +
        0.4 * address_features['address_token_jaccard']
    )

    return features


def compute_all_features(row1: Dict[str, Any], row2: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute all features for a pair of business records.

    Args:
        row1: Dict with keys 'business_name', 'business_address', 'country'
        row2: Dict with keys 'business_name', 'business_address', 'country'

    Returns:
        Dict of feature name -> value
    """
    name_feats = compute_name_features(row1['business_name'], row2['business_name'])
    addr_feats = compute_address_features(row1['business_address'], row2['business_address'])
    country_feats = compute_country_features(row1['country'], row2['country'])
    cross_feats = compute_cross_features(name_feats, addr_feats, country_feats)

    # Combine all features
    all_features = {}
    all_features.update(name_feats)
    all_features.update(addr_feats)
    all_features.update(country_feats)
    all_features.update(cross_feats)

    return all_features


def get_feature_names() -> list:
    """Return list of all feature names in order."""
    # Create dummy features to get the keys
    dummy_row = {'business_name': 'test', 'business_address': 'test', 'country': 'us'}
    features = compute_all_features(dummy_row, dummy_row)
    return sorted(features.keys())
