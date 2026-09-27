"""Configuration for entity resolution pipeline."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training files
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Test files
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Output files
MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"
MODEL_PATH = OUTPUT_DIR / "trained_model.joblib"
EXPERIMENT_LOG = OUTPUT_DIR / "experiments.csv"

# Random seed for reproducibility
RANDOM_SEED = 42

# Validation split ratio
VALIDATION_SPLIT = 0.15

# Candidate generation parameters
MAX_CANDIDATES_PER_ENTITY = 200
MIN_NAME_SIMILARITY_BLOCKING = 0.3
MIN_NGRAM_OVERLAP = 2

# Model parameters
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}

# Threshold search range
THRESHOLD_SEARCH_MIN = 0.1
THRESHOLD_SEARCH_MAX = 0.95
THRESHOLD_SEARCH_STEP = 0.05

# Parallel processing
N_JOBS = -1
