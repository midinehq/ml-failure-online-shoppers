from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import (
    BROKEN_FEATURES,
    CORRECTED_FEATURES,
    METRICS_DIR,
    NO_PAGEVALUE_FEATURES,
    SEEDS,
    SHORTCUT_FEATURES,
    TARGET,
)
from .evaluation import metric_row, subgroup_metrics
from .preprocessing import adjacent_temporal_split, random_reference_split, temporal_split
from .training import fit_predict


def evaluate_result(result):
    return metric_row(result["name"], result["y_test"], result["proba"], result["threshold"])


def run_reference_and_failure(df: pd.DataFrame, seed: int = 2026):
    random_train, random_valid, random_test = random_reference_split(df)
    temporal_train, temporal_valid, temporal_test = temporal_split(df)
    random_broken = fit_predict(
        random_train, random_valid, random_test, BROKEN_FEATURES, "broken_random_split", seed=seed
    )
    temporal_broken = fit_predict(
        temporal_train, temporal_valid, temporal_test, BROKEN_FEATURES, "broken_temporal_deployment", seed=seed
    )
    return random_broken, temporal_broken


def run_controlled_causal_test(df: pd.DataFrame, seed: int = 2026):
    train, valid, test = temporal_split(df)
    broken = fit_predict(train, valid, test, BROKEN_FEATURES, "shortcut_only", seed=seed)
    corrected = fit_predict(train, valid, test, CORRECTED_FEATURES, "behavioral_model", seed=seed)
    no_page = fit_predict(train, valid, test, NO_PAGEVALUE_FEATURES, "behavioral_without_pagevalues", seed=seed)
    return broken, corrected, no_page


def run_corrected_protocol(df: pd.DataFrame, seed: int = 2026):
    train, valid, test = adjacent_temporal_split(df)
    return fit_predict(
        train,
        valid,
        test,
        CORRECTED_FEATURES,
        "corrected_invariant_temporal_validation",
        seed=seed,
        threshold_objective="recall_at_precision_30",
        class_weight="balanced_subsample",
    )


def run_seed_variance(df: pd.DataFrame):
    rows = []
    train, valid, test = temporal_split(df)
    corrected_train, corrected_valid, corrected_test = adjacent_temporal_split(df)
    for seed in SEEDS:
        broken = fit_predict(train, valid, test, BROKEN_FEATURES, "broken_shortcut", seed=seed)
        fixed = fit_predict(
            corrected_train,
            corrected_valid,
            corrected_test,
            CORRECTED_FEATURES,
            "corrected_invariant",
            seed=seed,
            threshold_objective="recall_at_precision_30",
            class_weight="balanced_subsample",
        )
        rows.append(evaluate_result(broken))
        rows.append(evaluate_result(fixed))
    return pd.DataFrame(rows)


def feature_perturbation_test(result, feature: str, seed: int = 2026):
    """Intervene on one input column at prediction time and measure damage.

    This is causal with respect to the fitted prediction function: the trained
    model is held fixed and only one observed variable is randomized.
    """
    if feature not in result["features"]:
        raise ValueError(f"{feature} is not used by model {result['name']}")
    rng = np.random.default_rng(seed)
    frame = result["test_frame"].copy()
    perturbed = frame[result["features"]].copy()
    perturbed[feature] = rng.permutation(perturbed[feature].to_numpy())
    original = metric_row(
        f"{result['name']}_original",
        result["y_test"],
        result["proba"],
        result["threshold"],
    )
    perturbed_proba = result["pipeline"].predict_proba(perturbed)[:, 1]
    intervention = metric_row(
        f"{result['name']}_permute_{feature}",
        result["y_test"],
        perturbed_proba,
        result["threshold"],
    )
    rows = pd.DataFrame([original, intervention])
    rows["intervention_feature"] = feature
    rows["delta_pr_auc_vs_original"] = rows["pr_auc"] - original["pr_auc"]
    rows["delta_recall_vs_original"] = rows["recall"] - original["recall"]
    return rows


def negative_control_representation_test(df: pd.DataFrame, seed: int = 2026):
    """Compare an intended causal intervention with a negative-control representation.

    The corrected model excludes calendar/acquisition variables. If the repair is
    doing what we claim, perturbing Month should damage the shortcut model but
    should not be an available failure path in the corrected representation.
    """
    broken, behavioral, _ = run_controlled_causal_test(df, seed=seed)
    month_intervention = feature_perturbation_test(broken, "Month", seed=seed)
    browser_intervention = feature_perturbation_test(broken, "Browser", seed=seed)
    representation = pd.DataFrame(
        [
            {
                "test": "shortcut_model_uses_month",
                "result": "Month is present and can be intervened on",
                "model": broken["name"],
            },
            {
                "test": "corrected_model_excludes_month",
                "result": "Month is absent by design, so this causal path is blocked",
                "model": behavioral["name"],
            },
        ]
    )
    return month_intervention, browser_intervention, representation


def permutation_importance_table(result, n_repeats: int = 20):
    frame = result["test_frame"]
    x = frame[result["features"]]
    y = frame[TARGET]
    importance = permutation_importance(
        result["pipeline"],
        x,
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=2026,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": result["features"],
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def drift_table(df: pd.DataFrame):
    rows = []
    early = df[df["MonthIndex"] <= 9]
    late = df[df["MonthIndex"] >= 10]
    for col in SHORTCUT_FEATURES + CORRECTED_FEATURES:
        if col not in df.columns:
            continue
        if df[col].dtype == "object":
            early_dist = early[col].value_counts(normalize=True)
            late_dist = late[col].value_counts(normalize=True)
            cats = sorted(set(early_dist.index) | set(late_dist.index))
            distance = 0.5 * sum(abs(early_dist.get(c, 0) - late_dist.get(c, 0)) for c in cats)
        else:
            denom = df[col].std() or 1.0
            distance = abs(early[col].mean() - late[col].mean()) / denom
        rows.append({"feature": col, "shift_score": float(distance)})
    return pd.DataFrame(rows).sort_values("shift_score", ascending=False)


def write_metrics(name: str, df: pd.DataFrame):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / name
    df.to_csv(path, index=False)
    return path
