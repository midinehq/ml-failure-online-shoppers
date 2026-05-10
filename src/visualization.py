from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

from .config import FIGURES_DIR
from .evaluation import confusion

sns.set_theme(style="whitegrid", context="notebook")


def savefig(name: str, fig=None, directory: Path = FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    path = directory / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def plot_confusion(y_true, proba, threshold, title, filename):
    cm = confusion(y_true, proba, threshold)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    savefig(filename, fig)
    return fig, cm


def plot_subgroup(metric_df: pd.DataFrame, metric: str, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=metric_df, x="group", y=metric, hue="model", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="")
    savefig(filename, fig)
    return fig


def plot_metric_comparison(metrics_df: pd.DataFrame, filename: str):
    keep = ["model", "balanced_accuracy", "recall", "f1", "roc_auc", "pr_auc", "brier", "ece"]
    long = metrics_df[keep].melt("model", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=long, x="metric", y="value", hue="model", ax=ax)
    ax.set_title("Before/after comparison on late-year deployment months")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="")
    savefig(filename, fig)
    return fig

def plot_evidence_table(evidence: pd.DataFrame, filename: str):

    plot_df = evidence.copy()
    plot_df["model"] = plot_df["model"].str.strip()

    rename_map = {
        "broken_random_split": "random split",
        "broken_temporal_deployment": "temporal broken",
        "corrected_invariant_temporal_validation": "corrected"
    }
    plot_df["model"] = plot_df["model"].map(rename_map).fillna(plot_df["model"])

    plot_df["error_low"]  = plot_df["estimate"] - plot_df["ci_low"]
    plot_df["error_high"] = plot_df["ci_high"]  - plot_df["estimate"]

    metrics = plot_df["metric"].unique()
    model_order = ["random split", "temporal broken", "corrected"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4), sharey=False)  # ← sharey=False

    if len(metrics) == 1:
        axes = [axes]

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        sub = (
            plot_df[plot_df["metric"] == metric]
            .set_index("model")
            .reindex(model_order)
            .reset_index()
        )

        ax.errorbar(
            sub["estimate"],
            range(len(sub)),
            xerr=[sub["error_low"].values, sub["error_high"].values],
            fmt="o", capsize=4, linewidth=2, markersize=7,
        )

        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("estimate\n95% CI", fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["model"].values, fontsize=9)  # ← labels sur TOUS les axes

    fig.suptitle("Failure evidence table visualized as bootstrap intervals", fontsize=13, y=1.02)
    plt.tight_layout()
    savefig(filename, fig)
    return fig

def plot_drift_scores(drift_df: pd.DataFrame, filename: str, top_n: int = 12):
    sub = drift_df.head(top_n).sort_values("shift_score")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.barplot(data=sub, x="shift_score", y="feature", ax=ax, color="#4c78a8")
    ax.set_title("Largest train-deployment distribution shifts")
    ax.set_xlabel("Shift score")
    ax.set_ylabel("")
    savefig(filename, fig)
    return fig


def plot_permutation_importance(importance_df: pd.DataFrame, title: str, filename: str, top_n: int = 12):
    sub = importance_df.head(top_n).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(sub["feature"], sub["importance_mean"], xerr=sub["importance_std"], color="#59a14f")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Decrease in PR-AUC after permutation")
    savefig(filename, fig)
    return fig


def plot_learning_curve(train_sizes, train_scores, valid_scores, filename: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="train")
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.18,
    )
    ax.plot(train_sizes, valid_scores.mean(axis=1), marker="o", label="validation")
    ax.fill_between(
        train_sizes,
        valid_scores.mean(axis=1) - valid_scores.std(axis=1),
        valid_scores.mean(axis=1) + valid_scores.std(axis=1),
        alpha=0.18,
    )
    ax.set_title("Learning curve: capacity versus generalization")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("PR-AUC")
    ax.legend()
    savefig(filename, fig)
    return fig


def plot_calibration(y_true, proba_by_model: dict[str, list[float]], filename: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for name, proba in proba_by_model.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed purchase rate")
    ax.set_title("Calibration under temporal shift")
    ax.legend()
    savefig(filename, fig)
    return fig
