from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED, TARGET
from .preprocessing import make_preprocessor


def make_model(seed: int = RANDOM_SEED, class_weight=None, regularized: bool = True):
    max_depth = 6 if regularized else None
    min_leaf = 25 if regularized else 1
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        max_features="sqrt",
        class_weight=class_weight,
        n_jobs=-1,
        random_state=seed,
    )


def make_pipeline(x: pd.DataFrame, seed: int = RANDOM_SEED, class_weight=None, regularized=True):
    return Pipeline(
        [
            ("preprocess", make_preprocessor(x)),
            ("model", make_model(seed, class_weight=class_weight, regularized=regularized)),
        ]
    )


def choose_threshold(y_true, proba, objective: str = "f1") -> float:
    thresholds = np.linspace(0.02, 0.80, 160)
    if objective == "recall_at_precision_30":
        best_t, best_recall = 0.5, -1.0
        for threshold in thresholds:
            pred = proba >= threshold
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            if precision >= 0.30 and recall > best_recall:
                best_t, best_recall = threshold, recall
        return float(best_t)
    scores = [f1_score(y_true, proba >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def fit_predict(train, valid, test, features, name, seed=RANDOM_SEED, threshold_objective="f1", class_weight=None):
    x_train, y_train = train[features], train[TARGET]
    x_valid, y_valid = valid[features], valid[TARGET]
    x_test, y_test = test[features], test[TARGET]
    pipe = make_pipeline(x_train, seed=seed, class_weight=class_weight)
    pipe.fit(x_train, y_train)
    valid_proba = pipe.predict_proba(x_valid)[:, 1]
    threshold = choose_threshold(y_valid.to_numpy(), valid_proba, threshold_objective)
    test_proba = pipe.predict_proba(x_test)[:, 1]
    return {
        "name": name,
        "pipeline": pipe,
        "proba": test_proba,
        "threshold": threshold,
        "y_test": y_test.to_numpy(),
        "test_frame": test.copy(),
        "features": features,
    }
