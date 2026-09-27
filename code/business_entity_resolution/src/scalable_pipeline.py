"""Reproducible, streaming entity-resolution pipeline for the full challenge data."""

import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

from rapidfuzz import fuzz

from evaluate import evaluate_predictions
from normalization import (
    extract_numbers,
    normalize_address,
    normalize_country,
    normalize_name,
    normalize_transliterate,
)
from sqlite_blocking import SQLiteCandidateGenerator

logger = logging.getLogger(__name__)


def record_rows(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            yield {key: value or "" for key, value in row.items()}


def pair_score(left: dict, right: dict) -> float:
    left_name = normalize_name(left["business_name"])
    right_name = normalize_name(right["business_name"])
    left_ascii = normalize_transliterate(left_name)
    right_ascii = normalize_transliterate(right_name)
    name = max(
        fuzz.ratio(left_name, right_name),
        fuzz.token_set_ratio(left_name, right_name),
        fuzz.ratio(left_ascii, right_ascii),
    ) / 100.0
    left_address = normalize_address(left["business_address"])
    right_address = normalize_address(right["business_address"])
    address = fuzz.token_set_ratio(left_address, right_address) / 100.0
    number_overlap = bool(extract_numbers(left_address) & extract_numbers(right_address))
    country = normalize_country(left["country"]) == normalize_country(right["country"])
    if left_name and left_name == right_name:
        return 1.0
    if left_address and left_address == right_address and name >= 0.62:
        return 0.98
    score = 0.68 * name + 0.22 * address + 0.07 * float(number_overlap) + 0.03 * float(country)
    return score


def predict_entity(left: dict, candidate_ids: Set[str], targets: Dict[str, dict], threshold: float):
    scored = [(pair_score(left, targets[entity_id]), entity_id)
              for entity_id in candidate_ids if entity_id in targets]
    return {entity_id for score, entity_id in scored if score >= threshold}


def load_ground_truth(path: Path, limit: int = 0) -> Dict[str, Set[str]]:
    result = {}
    for index, row in enumerate(record_rows(path)):
        if limit and index >= limit:
            break
        result[row["source1_entity_id"]] = {
            value for value in row["matched_entity_ids"].split(",") if value
        }
    return result


def tune_threshold(source1_path: Path, target_index: SQLiteCandidateGenerator,
                   ground_truth: Dict[str, Set[str]], thresholds: Iterable[float]):
    rows = {row["entity_id"]: row for row in record_rows(source1_path)
            if row["entity_id"] in ground_truth}
    candidates = {}
    scored_by_entity = {}
    for entity_id, truth in ground_truth.items():
        row = rows[entity_id]
        ids = target_index.candidates_for(row["business_name"], row["business_address"], row["country"])
        candidates[entity_id] = ids
        targets = target_index.records_for(ids)
        scores = [(pair_score(row, targets[target_id]), target_id)
                  for target_id in ids if target_id in targets]
        scored_by_entity[entity_id] = scores
    results = []
    for threshold in thresholds:
        predictions = {
            entity_id: {target_id for score, target_id in scores if score >= threshold}
            for entity_id, scores in scored_by_entity.items()
        }
        metrics = evaluate_predictions(predictions, ground_truth)
        results.append((metrics["macro_f05"], threshold, metrics))
    results.sort(reverse=True, key=lambda value: value[0])
    return results[0], candidates


def run_inference(source1_path: Path, target_index: SQLiteCandidateGenerator,
                  matching_path: Path, candidate_path: Path, threshold: float) -> dict:
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"entities": 0, "candidates": 0, "matches": 0, "empty": 0}
    with open(matching_path, "w", encoding="utf-8", newline="") as matching, \
            open(candidate_path, "w", encoding="utf-8", newline="") as candidates:
        match_writer = csv.writer(matching, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidates, delimiter="\t", lineterminator="\n")
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for row in record_rows(source1_path):
            entity_id = row["entity_id"]
            ids = target_index.candidates_for(row["business_name"], row["business_address"], row["country"])
            targets = target_index.records_for(ids)
            matches = predict_entity(row, ids, targets, threshold)
            ordered_ids = sorted(ids)
            ordered_matches = sorted(matches)
            candidate_writer.writerow([entity_id, ",".join(ordered_ids)])
            match_writer.writerow([entity_id, ",".join(ordered_matches)])
            stats["entities"] += 1
            stats["candidates"] += len(ids)
            stats["matches"] += len(matches)
            stats["empty"] += not matches
    return stats