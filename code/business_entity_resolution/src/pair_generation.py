"""Generate training pairs from ground truth and candidates."""

import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple
import logging
from tqdm import tqdm
try:
    from .features import compute_all_features
except ImportError:
    from features import compute_all_features

logger = logging.getLogger(__name__)


def create_record_lookup(df: pd.DataFrame) -> Dict[str, Dict]:
    """Create a lookup dict from entity_id to record dict."""
    lookup = {}
    for idx, row in df.iterrows():
        lookup[row['entity_id']] = {
            'business_name': row['business_name'],
            'business_address': row['business_address'],
            'country': row['country']
        }
    return lookup


def generate_training_pairs(source1_df: pd.DataFrame,
                            source2_df: pd.DataFrame,
                            source3_df: pd.DataFrame,
                            ground_truth: Dict[str, Set[str]],
                            candidates: Dict[str, Set[str]],
                            negative_ratio: float = 2.0,
                            seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate training pairs with features and labels.

    Args:
        source1_df: Source 1 dataframe
        source2_df: Source 2 dataframe
        source3_df: Source 3 dataframe
        ground_truth: S1 ID -> set of true match IDs
        candidates: S1 ID -> set of candidate IDs
        negative_ratio: Ratio of negatives to positives
        seed: Random seed

    Returns:
        (features_df, labels_array)
    """
    logger.info("Generating training pairs...")

    np.random.seed(seed)

    # Create lookups
    s1_lookup = create_record_lookup(source1_df)
    s2_s3_lookup = create_record_lookup(pd.concat([source2_df, source3_df], ignore_index=True))

    positive_pairs = []
    negative_pairs = []

    # Generate positive pairs
    logger.info("Generating positive pairs...")
    for s1_id, true_matches in tqdm(ground_truth.items(), desc="Positive pairs"):
        if not true_matches:  # Skip singletons
            continue

        if s1_id not in s1_lookup:
            continue

        for match_id in true_matches:
            if match_id in s2_s3_lookup:
                positive_pairs.append((s1_id, match_id, 1))

    logger.info(f"  Generated {len(positive_pairs):,} positive pairs")

    # Generate negative pairs (from candidates)
    logger.info("Generating negative pairs...")
    target_negatives = int(len(positive_pairs) * negative_ratio)

    for s1_id in tqdm(list(ground_truth.keys()), desc="Negative pairs"):
        if s1_id not in candidates or s1_id not in s1_lookup:
            continue

        true_matches = ground_truth.get(s1_id, set())
        candidate_set = candidates[s1_id]

        # Hard negatives: candidates that are NOT true matches
        hard_negatives = candidate_set - true_matches

        if hard_negatives:
            # Sample negatives for this entity
            n_to_sample = min(len(hard_negatives), max(1, int(len(true_matches) * negative_ratio) + 1))
            sampled = np.random.choice(list(hard_negatives), size=n_to_sample, replace=False)

            for neg_id in sampled:
                if neg_id in s2_s3_lookup:
                    negative_pairs.append((s1_id, neg_id, 0))

                if len(negative_pairs) >= target_negatives:
                    break

        if len(negative_pairs) >= target_negatives:
            break

    logger.info(f"  Generated {len(negative_pairs):,} negative pairs")

    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    np.random.shuffle(all_pairs)

    logger.info(f"Total pairs: {len(all_pairs):,}")
    logger.info(f"Positive ratio: {len(positive_pairs)/len(all_pairs):.3f}")

    # Extract features
    logger.info("Extracting features...")
    feature_dicts = []
    labels = []

    for s1_id, match_id, label in tqdm(all_pairs, desc="Feature extraction"):
        s1_record = s1_lookup[s1_id]
        s2_s3_record = s2_s3_lookup[match_id]

        features = compute_all_features(s1_record, s2_s3_record)
        feature_dicts.append(features)
        labels.append(label)

    features_df = pd.DataFrame(feature_dicts)
    labels_array = np.array(labels)

    logger.info(f"Feature matrix shape: {features_df.shape}")
    logger.info(f"Positive samples: {labels_array.sum():,}")
    logger.info(f"Negative samples: {(1-labels_array).sum():,.0f}")

    return features_df, labels_array


def generate_inference_pairs(source1_df: pd.DataFrame,
                             source2_df: pd.DataFrame,
                             source3_df: pd.DataFrame,
                             candidates: Dict[str, Set[str]]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Generate inference pairs from candidates.

    Args:
        source1_df: Source 1 dataframe
        source2_df: Source 2 dataframe
        source3_df: Source 3 dataframe
        candidates: S1 ID -> set of candidate IDs

    Returns:
        (features_df, pair_ids) where pair_ids is list of (s1_id, candidate_id) tuples
    """
    logger.info("Generating inference pairs...")

    # Create lookups
    s1_lookup = create_record_lookup(source1_df)
    s2_s3_lookup = create_record_lookup(pd.concat([source2_df, source3_df], ignore_index=True))

    feature_dicts = []
    pair_ids = []

    for s1_id in tqdm(list(candidates.keys()), desc="Inference pairs"):
        if s1_id not in s1_lookup:
            continue

        candidate_set = candidates[s1_id]
        s1_record = s1_lookup[s1_id]

        for candidate_id in candidate_set:
            if candidate_id not in s2_s3_lookup:
                continue

            s2_s3_record = s2_s3_lookup[candidate_id]
            features = compute_all_features(s1_record, s2_s3_record)

            feature_dicts.append(features)
            pair_ids.append((s1_id, candidate_id))

    features_df = pd.DataFrame(feature_dicts)

    logger.info(f"Generated {len(pair_ids):,} inference pairs")
    logger.info(f"Feature matrix shape: {features_df.shape}")

    return features_df, pair_ids
