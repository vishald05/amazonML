"""Run streaming inference and write the two required submission files."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CANDIDATE_PAIRS, MATCHING_RESULTS, OUTPUT_DIR, TEST_DIR
from scalable_pipeline import run_inference
from sqlite_blocking import SQLiteCandidateGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    threshold = float((OUTPUT_DIR / "best_threshold.txt").read_text(encoding="utf-8").strip())
    index = SQLiteCandidateGenerator(OUTPUT_DIR / "test_target_index.sqlite", max_candidates=200)
    index.build([TEST_DIR / "test_source2.tsv", TEST_DIR / "test_source3.tsv"])
    stats = run_inference(
        TEST_DIR / "test_source1.tsv", index,
        MATCHING_RESULTS, CANDIDATE_PAIRS, threshold,
    )
    logging.info("Inference complete: %s", stats)
    index.close()


if __name__ == "__main__":
    main()
