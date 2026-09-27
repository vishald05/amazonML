"""Candidate generation (blocking) strategies."""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import logging
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing as mp
from functools import partial

try:
    from .normalization import normalize_name, normalize_address, get_tokens, get_char_ngrams
except ImportError:
    from normalization import normalize_name, normalize_address, get_tokens, get_char_ngrams

logger = logging.getLogger(__name__)


def process_chunk(chunk):
    """Process a chunk of records for indexing."""
    name_idx = defaultdict(set)
    token_idx = defaultdict(set)
    ngram_idx = defaultdict(set)
    combined_texts = []
    entity_ids = []

    for idx, row in chunk.iterrows():
        entity_id = row['entity_id']
        name = normalize_name(row['business_name'])
        address = normalize_address(row['business_address'])

        # Combined text for TF-IDF
        combined = f"{name} {address}".strip()
        combined_texts.append(combined)
        entity_ids.append(entity_id)

        # Exact name
        if name:
            name_idx[name].add(entity_id)

        # Tokens
        for token in get_tokens(name):
            if len(token) >= 3:
                token_idx[token].add(entity_id)

        # N-grams (sample to avoid huge index memory)
        ngrams = get_char_ngrams(name, 3)
        if len(ngrams) > 10:
            # If too many n-grams, keep a determinisitic subset to save memory
            ngrams = sorted(list(ngrams))
            ngrams = {ngrams[i] for i in range(0, len(ngrams), max(1, len(ngrams)//10))}

        for ngram in ngrams:
            ngram_idx[ngram].add(entity_id)

    return name_idx, token_idx, ngram_idx, combined_texts, entity_ids


class CandidateGenerator:
    """Multi-stage candidate generation for entity resolution."""

    def __init__(self, max_candidates_per_entity: int = 200, n_jobs=-1):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.name_index = defaultdict(set)
        self.token_index = defaultdict(set)
        self.ngram_index = defaultdict(set)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.entity_ids_list = []
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()

    def build_indices(self, source2_df: pd.DataFrame, source3_df: pd.DataFrame):
        """Build inverted indices for candidate generation efficiently."""
        logger.info(f"Building indices using {self.n_jobs} processes...")

        all_records = pd.concat([source2_df, source3_df], ignore_index=True)
        logger.info(f"  Total records to index: {len(all_records):,}")

        # Split data into chunks
        chunk_size = max(10000, len(all_records) // (self.n_jobs * 4))
        chunks = [all_records.iloc[i:i + chunk_size] for i in range(0, len(all_records), chunk_size)]

        logger.info(f"  Processing {len(chunks)} chunks in parallel...")

        # Process chunks in parallel
        with mp.Pool(self.n_jobs) as pool:
            results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Building indices"))

        # Merge results
        logger.info("Merging indices...")
        combined_texts = []

        for name_idx, token_idx, ngram_idx, texts, e_ids in tqdm(results, desc="Merging"):
            for k, v in name_idx.items():
                self.name_index[k].update(v)
            for k, v in token_idx.items():
                self.token_index[k].update(v)
            for k, v in ngram_idx.items():
                self.ngram_index[k].update(v)

            combined_texts.extend(texts)
            self.entity_ids_list.extend(e_ids)

        logger.info(f"  Name index size: {len(self.name_index):,}")
        logger.info(f"  Token index size: {len(self.token_index):,}")
        logger.info(f"  N-gram index size: {len(self.ngram_index):,}")

        # Build limited TF-IDF to save memory and time
        logger.info("Building TF-IDF index (limited features)...")
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), # Reduce from (1,3) to save time/mem
            analyzer='char_wb',
            max_features=2500,  # Reduce max features
            min_df=5
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_texts)
        logger.info(f"  TF-IDF matrix shape: {self.tfidf_matrix.shape}")


    def generate_candidates(self, source1_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Generate candidate matches for all Source 1 entities."""
        logger.info(f"Generating candidates for {len(source1_df):,} Source 1 entities...")

        all_candidates = {}
        for idx, row in tqdm(source1_df.iterrows(), total=len(source1_df), desc="Candidate generation"):
            all_candidates[row['entity_id']] = self._get_candidates_for_entity(row)

        self._print_stats(all_candidates)
        return all_candidates

    def _print_stats(self, all_candidates):
        total_candidates = sum(len(c) for c in all_candidates.values())
        avg_candidates = total_candidates / max(1, len(all_candidates))
        max_candidates = max(len(c) for c in all_candidates.values()) if all_candidates else 0
        zero_candidates = sum(1 for c in all_candidates.values() if len(c) == 0)

        logger.info(f"  Total candidates: {total_candidates:,}")
        logger.info(f"  Average per entity: {avg_candidates:.1f}")
        logger.info(f"  Max per entity: {max_candidates}")
        logger.info(f"  Entities with zero candidates: {zero_candidates:,}")

    def _get_candidates_for_entity(self, s1_row: pd.Series) -> Set[str]:
        """Generate candidates for a single Source 1 entity."""
        candidates = set()

        name = normalize_name(s1_row['business_name'])
        if not name:
            # If no name, try with address and TF-IDF fallback
            return self._tfidf_nearest_neighbors(normalize_address(s1_row['business_address']), k=10)

        # Strategy 1: Exact normalized name match
        if name in self.name_index:
            candidates.update(self.name_index[name])

        if len(candidates) >= self.max_candidates_per_entity:
            return set(list(candidates)[:self.max_candidates_per_entity])

        # Strategy 2: Token-based blocking
        name_tokens = get_tokens(name)
        if name_tokens:
            min_token_overlap = 1 if len(name_tokens) <= 2 else 2
            token_candidates = defaultdict(int)

            for token in name_tokens:
                if token in self.token_index:
                    for cid in self.token_index[token]:
                        token_candidates[cid] += 1

            for cid, overlap in token_candidates.items():
                if overlap >= min_token_overlap:
                    candidates.add(cid)

        if len(candidates) >= self.max_candidates_per_entity:
            return set(list(candidates)[:self.max_candidates_per_entity])

        # Strategy 3: N-gram fallback for very few candidates
        if len(candidates) < 5:
            ngrams = set(get_char_ngrams(name, 3))
            if ngrams:
                ngram_candidates = defaultdict(int)
                for ngram in ngrams:
                    if ngram in self.ngram_index:
                        for cid in self.ngram_index[ngram]:
                            ngram_candidates[cid] += 1

                threshold = max(2, int(len(ngrams) * 0.3))
                for cid, overlap in ngram_candidates.items():
                    if overlap >= threshold:
                        candidates.add(cid)

        # Truncate
        if len(candidates) > self.max_candidates_per_entity:
            candidates = set(list(candidates)[:self.max_candidates_per_entity])

        return candidates

    def _tfidf_nearest_neighbors(self, query_text: str, k: int = 10) -> Set[str]:
        """Find nearest neighbors using TF-IDF."""
        if not query_text or self.tfidf_matrix is None:
            return set()
        try:
            vec = self.tfidf_vectorizer.transform([query_text])
            similarities = cosine_similarity(vec, self.tfidf_matrix).flatten()
            top_k_indices = np.argsort(similarities)[-k:]
            return {self.entity_ids_list[i] for i in top_k_indices if similarities[i] > 0.3}
        except:
            return set()


def evaluate_candidate_recall(candidates: Dict[str, Set[str]],
                              ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """Evaluate candidate generation recall."""
    total_true_matches = sum(len(matches) for matches in ground_truth.values())
    found_matches = sum(
        sum(1 for tm in true_matches if tm in candidates.get(s1_id, set()))
        for s1_id, true_matches in ground_truth.items()
    )

    metrics = {
        'candidate_recall': found_matches / total_true_matches if total_true_matches > 0 else 0.0,
        'total_true_matches': total_true_matches,
        'found_matches': found_matches,
        'entities_evaluated': sum(1 for s1 in ground_truth.keys() if s1 in candidates)
    }
    return metrics
