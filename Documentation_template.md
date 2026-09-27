# ML Challenge 2026: Business Entity Resolution

**Team:** Vishal Team  
**Date:** 2026-09-27

## Methodology

The supplied TSV files contain 2,206,821 Source 1 records, 5,034,616 Source 2 records, and 5,285,603 Source 3 records. The ground truth contains 7,638,365 links and 123,247 singleton Source 1 entities. Country is treated as an open string feature; no country list is hard-coded.

The pipeline uses conservative Unicode/punctuation/whitespace normalization, address abbreviation normalization, transliterated name comparison, token overlap, numeric address overlap, and country equality. It supports zero, one, or many matches per Source 1 entity.

## Candidate Generation

Target records are streamed into an on-disk SQLite index. Blocking is the deterministic union of normalized name, transliterated name, normalized address, first name token, and first two sorted name tokens. Candidates are ranked by the number of agreeing blocks and capped at 200 per Source 1 entity. The exact final candidate set is written to `output/candidate_pairs.tsv`.

This design avoids the previous multiprocessing/global TF-IDF memory spike. Candidate recall is measured against the labeled tuning sample before inference.

## Matching and Validation

Each candidate receives a deterministic similarity score from normalized/transliterated name similarity, token-set address similarity, numeric overlap, country equality, and exact-name/address rules. The threshold is selected on a deterministic 100-entity training calibration slice using the same blocking and scoring code used at test time. This small slice is not an unbiased leaderboard estimate. No test labels or external business data are used.

The recorded calibration result is macro F0.5 0.4353, precision 0.4923, recall 0.4430, candidate recall 0.4898, threshold 0.80. Metrics are recorded in `output/training_metrics.json` and `output/experiments.csv`.

## Reproduction

From the repository root:

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
PYTHONPATH=code/business_entity_resolution/src python code/business_entity_resolution/src/train.py
PYTHONPATH=code/business_entity_resolution/src python code/business_entity_resolution/src/inference.py
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
python3 code/business_entity_resolution/src/create_submission.py VishalTeam
```

`train.py` builds a temporary training index, tunes and saves `output/best_threshold.txt`, and records metrics. `inference.py` builds a separate test index and streams one output row for every test Source 1 entity. Full test inference was not completed in this environment, so the required TSVs and final ZIP remain pending. SQLite indexes are temporary and excluded from the ZIP.

## Limitations

The blocking keys prioritize memory and precision and can miss matches whose names and addresses share no retained token or exact key. The tuning sample is deterministic and bounded rather than a full holdout; reported metrics should therefore be interpreted as a reproducible calibration result, not an unbiased leaderboard estimate. The final test includes France, while labeled training countries are US and India, so country-independent text signals are important.
