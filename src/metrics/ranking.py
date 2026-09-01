"""Threshold-free ranking metrics.

Both are tie-aware. They depend only on the scores, never on the decision rule,
so changing the threshold cannot move them.
"""

from __future__ import annotations

import numpy as np


def binary_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1.0e-12)
    accuracy = (tp + tn) / max(len(labels), 1)
    false_positive_rate = fp / max(fp + tn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "false_positive_rate": false_positive_rate,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "auroc": roc_auc(labels, scores),
        "aupr": average_precision(labels, scores),
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        # 1-based average rank gives every tied item the same contribution.
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    cumulative_fp = np.cumsum(sorted_labels == 0)
    # Evaluate the precision-recall curve only after a complete tie group.
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    true_positives = cumulative_tp[group_ends].astype(np.float64)
    false_positives = cumulative_fp[group_ends].astype(np.float64)
    recall = true_positives / positives
    precision = true_positives / (true_positives + false_positives)
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))
