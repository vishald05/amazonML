"""Input/output utilities for loading and saving data."""

import pandas as pd
from typing import Dict, Set, List, Tuple
import logging

logger = logging.getLogger(__name__)


def load_source_data(path, source_name="source"):
    """Load a TSV source file."""
    logger.info(f"Loading {source_name} from {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.fillna("")
    logger.info(f"  Loaded {len(df):,} records")
    return df


def load_ground_truth(path) -> Dict[str, Set[str]]:
    """
    Load ground truth file and return mapping: S1 ID -> set of matched IDs.
    """
    logger.info(f"Loading ground truth from {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)

    ground_truth = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row['matched_entity_ids']
        if pd.isna(matched) or matched == "":
            ground_truth[s1_id] = set()
        else:
            ground_truth[s1_id] = set(matched.split(','))

    logger.info(f"  Loaded ground truth for {len(ground_truth):,} S1 entities")
    total_matches = sum(len(v) for v in ground_truth.values())
    singletons = sum(1 for v in ground_truth.values() if len(v) == 0)
    logger.info(f"  Total matches: {total_matches:,}")
    logger.info(f"  Singletons: {singletons:,} ({100*singletons/len(ground_truth):.1f}%)")

    return ground_truth


def save_matching_results(results: Dict[str, List[str]], output_path):
    """
    Save matching results to TSV format.

    Args:
        results: Dict mapping S1 ID -> list of matched IDs
        output_path: Output file path
    """
    logger.info(f"Saving matching results to {output_path}")

    rows = []
    for s1_id in sorted(results.keys()):
        matched_ids = results[s1_id]
        if matched_ids:
            matched_str = ",".join(matched_ids)
        else:
            matched_str = ""
        rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': matched_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)

    total_matches = sum(len(v) for v in results.values())
    empty = sum(1 for v in results.values() if len(v) == 0)
    logger.info(f"  Saved {len(results):,} S1 entities")
    logger.info(f"  Total matches: {total_matches:,}")
    logger.info(f"  Empty predictions: {empty:,}")


def save_candidate_pairs(candidates: Dict[str, List[str]], output_path):
    """
    Save candidate pairs to TSV format.

    Args:
        candidates: Dict mapping S1 ID -> list of candidate IDs
        output_path: Output file path
    """
    logger.info(f"Saving candidate pairs to {output_path}")

    rows = []
    for s1_id in sorted(candidates.keys()):
        candidate_ids = candidates[s1_id]
        if candidate_ids:
            candidate_str = ",".join(candidate_ids)
        else:
            candidate_str = ""
        rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': candidate_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)

    total_candidates = sum(len(v) for v in candidates.values())
    logger.info(f"  Saved {len(candidates):,} S1 entities")
    logger.info(f"  Total candidates: {total_candidates:,}")
    logger.info(f"  Avg candidates per entity: {total_candidates/max(len(candidates), 1):.1f}")
