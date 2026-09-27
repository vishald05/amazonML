# Business Entity Resolution

This package resolves every Source 1 test record to zero or more Source 2/3 records using only the supplied challenge data.

## Setup

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

The data must remain under `dataset/train/` and `dataset/test/`. No external business data is used.

## Reproduce

Run from the repository root:

```bash
PYTHONPATH=code/business_entity_resolution/src python code/business_entity_resolution/src/train.py
PYTHONPATH=code/business_entity_resolution/src python code/business_entity_resolution/src/inference.py
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Training streams the target sources into a temporary SQLite index, tunes a precision-oriented threshold on a deterministic 100-entity labeled calibration slice, and records metrics in `output/training_metrics.json` and `output/experiments.csv`. Inference builds a separate test index from test Source 2/3 and streams both required output files. The index is temporary and is not required in the submission ZIP.

## Pipeline

1. Normalize names, addresses, countries, and transliterated names while preserving address numbers.
2. Block on normalized name, transliterated name, normalized address, and deterministic name-token keys.
3. Rank and cap candidates at 200 per Source 1 entity.
4. Score each candidate using name similarity, address token similarity, numeric overlap, country equality, and exact-match rules.
5. Keep all candidates above the tuned threshold, including multiple matches, or emit an empty list for a singleton.

The evaluator is macro-averaged per Source 1 entity and uses the challenge F0.5 definition. Candidate recall is measured before thresholding. Countries are open-set strings; France and any future country values are accepted without code changes.

## Output

`output/matching_results.tsv` contains exactly one row for every test Source 1 ID. `output/candidate_pairs.tsv` is the final candidate set passed to scoring, so every emitted match is a candidate. Both files are tab-separated and use comma-separated ID lists.

## Packaging

After validation succeeds:

```bash
python3 code/business_entity_resolution/src/create_submission.py VishalTeam
```

This creates `VishalTeam_submission.zip` with the required code, outputs, and documentation, excluding datasets, indexes, caches, and temporary logs.
