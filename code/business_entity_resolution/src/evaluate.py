"""Evaluation metrics for entity resolution."""

import numpy as np
from typing import Dict, Set, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F-beta score."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def evaluate_entity(predicted: Set[str], true: Set[str]) -> Tuple[float, float, float]:
    """
    Evaluate a single entity's predictions.

    Returns:
        (precision, recall, f0.5)
    """
    if not predicted and not true:
        # Both empty - perfect match
        return 1.0, 1.0, 1.0

    if not predicted:
        # Predicted nothing
        if not true:
            return 1.0, 1.0, 1.0  # Correct singleton
        else:
            return 0.0, 0.0, 0.0  # Missed matches

    if not true:
        # True singleton but predicted matches - false positives
        return 0.0, 0.0, 0.0

    # Both non-empty
    tp = len(predicted & true)
    fp = len(predicted - true)
    fn = len(true - predicted)

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(true) if true else 0.0
    f05 = compute_f_beta(precision, recall, beta=0.5)

    return precision, recall, f05


def evaluate_predictions(predictions: Dict[str, Set[str]],
                        ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Evaluate predictions using macro-averaged F0.5.

    Args:
        predictions: S1 ID -> set of predicted match IDs
        ground_truth: S1 ID -> set of true match IDs

    Returns:
        Dict with evaluation metrics
    """
    precisions = []
    recalls = []
    f05_scores = []

    tp_total = 0
    fp_total = 0
    fn_total = 0

    singleton_correct = 0
    singleton_total = 0

    for s1_id in ground_truth.keys():
        true_matches = ground_truth[s1_id]
        pred_matches = predictions.get(s1_id, set())

        precision, recall, f05 = evaluate_entity(pred_matches, true_matches)

        precisions.append(precision)
        recalls.append(recall)
        f05_scores.append(f05)

        # Track confusion matrix
        if pred_matches and true_matches:
            tp = len(pred_matches & true_matches)
            fp = len(pred_matches - true_matches)
            fn = len(true_matches - pred_matches)
            tp_total += tp
            fp_total += fp
            fn_total += fn
        elif pred_matches and not true_matches:
            fp_total += len(pred_matches)
        elif not pred_matches and true_matches:
            fn_total += len(true_matches)

        # Singleton tracking
        if not true_matches:
            singleton_total += 1
            if not pred_matches:
                singleton_correct += 1

    # Macro averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f05 = np.mean(f05_scores)

    # Micro averages
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f05 = compute_f_beta(micro_precision, micro_recall, beta=0.5)

    # Singleton accuracy
    singleton_accuracy = singleton_correct / singleton_total if singleton_total > 0 else 0.0

    results = {
        'macro_f05': macro_f05,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'micro_f05': micro_f05,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'singleton_accuracy': singleton_accuracy,
        'singleton_correct': singleton_correct,
        'singleton_total': singleton_total,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'num_entities': len(ground_truth)
    }

    return results


def print_evaluation_results(results: Dict[str, float]):
    """Pretty print evaluation results."""
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Macro F0.5:       {results['macro_f05']:.4f}")
    logger.info(f"Macro Precision:  {results['macro_precision']:.4f}")
    logger.info(f"Macro Recall:     {results['macro_recall']:.4f}")
    logger.info("-" * 60)
    logger.info(f"Micro F0.5:       {results['micro_f05']:.4f}")
    logger.info(f"Micro Precision:  {results['micro_precision']:.4f}")
    logger.info(f"Micro Recall:     {results['micro_recall']:.4f}")
    logger.info("-" * 60)
    logger.info(f"Singleton Accuracy: {results['singleton_accuracy']:.4f}")
    logger.info(f"  Correct: {results['singleton_correct']:,} / {results['singleton_total']:,}")
    logger.info("-" * 60)
    logger.info(f"TP: {results['tp']:,}")
    logger.info(f"FP: {results['fp']:,}")
    logger.info(f"FN: {results['fn']:,}")
    logger.info(f"Total entities: {results['num_entities']:,}")
    logger.info("=" * 60)
