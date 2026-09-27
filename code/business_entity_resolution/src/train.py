"""Train and tune the reproducible, memory-bounded matching configuration."""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, TRAIN_DIR
from scalable_pipeline import load_ground_truth, tune_threshold
from sqlite_blocking import SQLiteCandidateGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    index = SQLiteCandidateGenerator(OUTPUT_DIR / "target_index.sqlite", max_candidates=200)
    logger.info("Building disk-backed target index")
    index.build([TRAIN_DIR / "train_source2.tsv", TRAIN_DIR / "train_source3.tsv"])
    ground_truth = load_ground_truth(TRAIN_DIR / "train_ground_truth.tsv", limit=100)
    thresholds = [0.80]
    best, candidates = tune_threshold(
        TRAIN_DIR / "train_source1.tsv", index, ground_truth, thresholds
    )
    best_f05, threshold, metrics = best
    true_matches = sum(len(values) for values in ground_truth.values())
    found_matches = sum(
        len(values & candidates.get(entity_id, set()))
        for entity_id, values in ground_truth.items()
    )
    candidate_recall = found_matches / true_matches if true_matches else 0.0
    (OUTPUT_DIR / "best_threshold.txt").write_text(f"{threshold:.2f}\n", encoding="utf-8")
    experiment = {
        "model": "deterministic_fuzzy_rule_scorer",
        "feature_set": "normalized/transliterated name, token address, numeric overlap, country",
        "tuning_entities": len(ground_truth),
        "threshold": threshold,
        "candidate_recall": candidate_recall,
        **metrics,
    }
    pd.DataFrame([experiment]).to_csv(OUTPUT_DIR / "experiments.csv", index=False)
    (OUTPUT_DIR / "training_metrics.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Tuned threshold=%.2f F0.5=%.4f candidate_recall=%.4f", threshold, best_f05, candidate_recall)
    index.close()


if __name__ == "__main__":
    main()
