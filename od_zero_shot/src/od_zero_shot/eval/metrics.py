"""指标计算与聚合。"""

from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score


def _safe_metric(func, *args, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return float("nan")


def binary_edge_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray, threshold: float) -> dict[str, float]:
    true_edge = (np.expm1(y_true_log) > 0.0).astype(np.int32).reshape(-1)
    pred_flow = np.expm1(y_pred_log).reshape(-1)
    pred_score = y_pred_log.reshape(-1)
    pred_edge = (pred_flow > threshold).astype(np.int32)
    return {
        "auroc": _safe_metric(roc_auc_score, true_edge, pred_score),
        "auprc": _safe_metric(average_precision_score, true_edge, pred_score),
        "f1_at_tau": _safe_metric(f1_score, true_edge, pred_edge, zero_division=0),
    }


def grouped_regression_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    mask_diag: np.ndarray,
    mask_pos_off: np.ndarray,
    mask_zero_off: np.ndarray,
) -> dict[str, float]:
    metrics = {}
    groups = {
        "all_pairs": np.ones_like(mask_diag, dtype=bool),
        "positive_off_diagonal": np.asarray(mask_pos_off).astype(bool),
        "diagonal_only": np.asarray(mask_diag).astype(bool),
        "zero_off_diagonal": np.asarray(mask_zero_off).astype(bool),
    }
    for name, mask in groups.items():
        if np.sum(mask) == 0:
            metrics[f"{name}_mae"] = float("nan")
            metrics[f"{name}_rmse"] = float("nan")
            continue
        truth = y_true_log[mask]
        pred = y_pred_log[mask]
        metrics[f"{name}_mae"] = float(mean_absolute_error(truth, pred))
        metrics[f"{name}_rmse"] = float(math.sqrt(mean_squared_error(truth, pred)))
    return metrics


def flow_conservation_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    true_flow = np.expm1(y_true_log)
    pred_flow = np.maximum(np.expm1(y_pred_log), 0.0)
    return {
        "row_sum_mae": float(mean_absolute_error(true_flow.sum(axis=1), pred_flow.sum(axis=1))),
        "column_sum_mae": float(mean_absolute_error(true_flow.sum(axis=0), pred_flow.sum(axis=0))),
        "total_flow_error": float(abs(true_flow.sum() - pred_flow.sum())),
    }


def top_k_recall(y_true_log: np.ndarray, y_pred_log: np.ndarray, top_k: int) -> float:
    true_flow = np.expm1(y_true_log).copy()
    pred_flow = np.expm1(y_pred_log).copy()
    np.fill_diagonal(true_flow, -np.inf)
    np.fill_diagonal(pred_flow, -np.inf)
    recalls = []
    for origin in range(true_flow.shape[0]):
        true_idx = np.argsort(true_flow[origin])[-top_k:]
        pred_idx = np.argsort(pred_flow[origin])[-top_k:]
        true_set = set(int(idx) for idx in true_idx if np.isfinite(true_flow[origin, idx]))
        pred_set = set(int(idx) for idx in pred_idx if np.isfinite(pred_flow[origin, idx]))
        if true_set:
            recalls.append(len(true_set & pred_set) / len(true_set))
    return float(np.mean(recalls)) if recalls else float("nan")


def degree_distribution_error(y_true_log: np.ndarray, y_pred_log: np.ndarray, threshold: float) -> dict[str, float]:
    true_adj = (np.expm1(y_true_log) > 0.0).astype(np.int32)
    pred_adj = (np.expm1(y_pred_log) > threshold).astype(np.int32)
    return {
        "out_degree_distribution_error": float(np.mean(np.abs(np.sort(true_adj.sum(axis=1)) - np.sort(pred_adj.sum(axis=1))))),
        "in_degree_distribution_error": float(np.mean(np.abs(np.sort(true_adj.sum(axis=0)) - np.sort(pred_adj.sum(axis=0))))),
    }


def distance_decay_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    distance_matrix_km: np.ndarray,
    bins: list[float],
) -> dict[str, float | dict[str, list[float]]]:
    valid_mask = np.ones_like(distance_matrix_km, dtype=bool)
    np.fill_diagonal(valid_mask, False)
    distances = distance_matrix_km[valid_mask]
    true_flow = np.expm1(y_true_log)[valid_mask]
    pred_flow = np.maximum(np.expm1(y_pred_log), 0.0)[valid_mask]
    if distances.size == 0:
        return {"distance_bin_curve_error": float("nan"), "distance_decay_curve": {"true_curve": [], "pred_curve": []}}
    edges = np.asarray(bins, dtype=np.float64)
    errors = []
    true_curve: list[float] = []
    pred_curve: list[float] = []
    for idx in range(len(edges) - 1):
        mask = (distances >= edges[idx]) & (distances <= edges[idx + 1] if idx == len(edges) - 2 else distances < edges[idx + 1])
        if np.sum(mask) == 0:
            true_curve.append(0.0)
            pred_curve.append(0.0)
            continue
        true_mean = float(true_flow[mask].mean())
        pred_mean = float(pred_flow[mask].mean())
        true_curve.append(true_mean)
        pred_curve.append(pred_mean)
        errors.append(abs(true_mean - pred_mean))
    return {
        "distance_bin_curve_error": float(np.mean(errors)) if errors else float("nan"),
        "distance_decay_curve": {"true_curve": true_curve, "pred_curve": pred_curve},
    }


def compute_all_metrics(sample: dict[str, Any], pred_log: np.ndarray, threshold: float, top_k: int, distance_bins: list[float]) -> dict[str, Any]:
    y_true = sample["y_od"]
    metrics: dict[str, Any] = {}
    metrics.update(binary_edge_metrics(y_true, pred_log, threshold))
    metrics.update(grouped_regression_metrics(y_true, pred_log, sample["mask_diag"], sample["mask_pos_off"], sample["mask_zero_off"]))
    metrics.update(flow_conservation_metrics(y_true, pred_log))
    metrics["top_k_recall"] = top_k_recall(y_true, pred_log, top_k=top_k)
    metrics.update(degree_distribution_error(y_true, pred_log, threshold))
    metrics.update(distance_decay_metrics(y_true, pred_log, sample["distance_matrix"], bins=distance_bins))
    return metrics


def aggregate_metrics(per_sample_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """对数值指标做跨样本聚合。"""

    numeric_keys = sorted(
        {
            key
            for row in per_sample_metrics
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "sample_id"
        }
    )
    aggregate: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in per_sample_metrics if isinstance(row.get(key), (int, float)) and not math.isnan(float(row[key]))]
        if not values:
            aggregate[key] = {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
            continue
        aggregate[key] = {
            "mean": float(mean(values)),
            "std": float(pstdev(values)) if len(values) > 1 else 0.0,
            "median": float(median(values)),
        }
    return aggregate
