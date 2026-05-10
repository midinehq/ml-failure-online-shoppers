from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import TARGET


def expected_calibration_error(y_true, proba, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    cuts = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(cuts[:-1], cuts[1:]):
        mask = (proba >= low) & (proba <= high if high == 1 else proba < high)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - proba[mask].mean())
    return float(ece)


def metric_row(name, y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
    }


def scalar_metric(metric: str, y_true, proba, threshold):
    pred = (np.asarray(proba) >= threshold).astype(int)
    y_true = np.asarray(y_true)
    if metric == "accuracy":
        return accuracy_score(y_true, pred)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, pred)
    if metric == "precision":
        return precision_score(y_true, pred, zero_division=0)
    if metric == "recall":
        return recall_score(y_true, pred, zero_division=0)
    if metric == "f1":
        return f1_score(y_true, pred, zero_division=0)
    if metric == "roc_auc":
        return roc_auc_score(y_true, proba)
    if metric == "pr_auc":
        return average_precision_score(y_true, proba)
    if metric == "brier":
        return brier_score_loss(y_true, proba)
    if metric == "ece":
        return expected_calibration_error(y_true, proba)
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_ci(y_true, proba, threshold, metric: str, n_boot: int = 1000, seed: int = 2026):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if metric in {"roc_auc", "pr_auc"} and len(np.unique(y_true[idx])) < 2:
            continue
        values.append(scalar_metric(metric, y_true[idx], proba[idx], threshold))
    values = np.asarray(values)
    return {
        "metric": metric,
        "estimate": scalar_metric(metric, y_true, proba, threshold),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n_boot": int(len(values)),
    }


def paired_bootstrap_delta(
    y_true,
    proba_before,
    threshold_before,
    proba_after,
    threshold_after,
    metric: str,
    n_boot: int = 1000,
    seed: int = 2026,
):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba_before = np.asarray(proba_before)
    proba_after = np.asarray(proba_after)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if metric in {"roc_auc", "pr_auc"} and len(np.unique(y_true[idx])) < 2:
            continue
        before = scalar_metric(metric, y_true[idx], proba_before[idx], threshold_before)
        after = scalar_metric(metric, y_true[idx], proba_after[idx], threshold_after)
        values.append(after - before)
    values = np.asarray(values)
    return {
        "metric": metric,
        "delta_after_minus_before": scalar_metric(metric, y_true, proba_after, threshold_after)
        - scalar_metric(metric, y_true, proba_before, threshold_before),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n_boot": int(len(values)),
    }


def evidence_table(results, metrics=("pr_auc", "recall", "balanced_accuracy", "ece", "brier"), n_boot=1000):
    rows = []
    for result in results:
        for metric in metrics:
            row = bootstrap_ci(
                result["y_test"],
                result["proba"],
                result["threshold"],
                metric,
                n_boot=n_boot,
            )
            row["model"] = result["name"]
            row["threshold"] = result["threshold"]
            row["n"] = len(result["y_test"])
            rows.append(row)
    columns = ["model", "metric", "estimate", "ci_low", "ci_high", "threshold", "n", "n_boot"]
    return pd.DataFrame(rows)[columns]


def quantitative_claims(random_result, temporal_result, corrected_result):
    random_metrics = metric_row(
        random_result["name"], random_result["y_test"], random_result["proba"], random_result["threshold"]
    )
    temporal_metrics = metric_row(
        temporal_result["name"], temporal_result["y_test"], temporal_result["proba"], temporal_result["threshold"]
    )
    corrected_metrics = metric_row(
        corrected_result["name"], corrected_result["y_test"], corrected_result["proba"], corrected_result["threshold"]
    )
    claims = []
    for metric, label in [
        ("pr_auc", "PR-AUC"),
        ("recall", "purchase recall"),
        ("balanced_accuracy", "balanced accuracy"),
        ("ece", "expected calibration error"),
    ]:
        claims.append(
            f"{label} changed from {random_metrics[metric]:.3f} under random validation "
            f"to {temporal_metrics[metric]:.3f} under temporal deployment."
        )
        claims.append(
            f"After correction, {label} was {corrected_metrics[metric]:.3f} on the same late-year deployment months."
        )
    return pd.DataFrame({"claim": claims})


def confusion(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return pd.DataFrame(
        confusion_matrix(y_true, pred),
        index=["actual_no_purchase", "actual_purchase"],
        columns=["pred_no_purchase", "pred_purchase"],
    )


def subgroup_metrics(frame, proba, threshold, group_col: str, model_name: str):
    rows = []
    for group, idx in frame.groupby(group_col).groups.items():
        y = frame.loc[idx, TARGET].to_numpy()
        p = proba[frame.index.get_indexer(idx)]
        if len(np.unique(y)) < 2:
            roc = np.nan
            pr = np.nan
        else:
            roc = roc_auc_score(y, p)
            pr = average_precision_score(y, p)
        pred = (p >= threshold).astype(int)
        rows.append(
            {
                "model": model_name,
                "group": group,
                "n": len(y),
                "prevalence": float(np.mean(y)),
                "recall": recall_score(y, pred, zero_division=0),
                "precision": precision_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "roc_auc": roc,
                "pr_auc": pr,
            }
        )
    return pd.DataFrame(rows).sort_values("group")


def summarize_runs(rows):
    df = pd.DataFrame(rows)
    metrics = ["balanced_accuracy", "recall", "f1", "roc_auc", "pr_auc", "brier", "ece"]
    return df.groupby("model")[metrics].agg(["mean", "std"]).round(4)
