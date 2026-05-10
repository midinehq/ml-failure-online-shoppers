from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 2026
SEEDS = [7, 13, 29, 43, 71]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

MONTH_ORDER = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "June": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

TARGET = "Revenue"

BEHAVIORAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
]

LEAKAGE_SENSITIVE_FEATURES = ["PageValues"]

SHORTCUT_FEATURES = [
    "Month",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

BROKEN_FEATURES = SHORTCUT_FEATURES
CORRECTED_FEATURES = BEHAVIORAL_FEATURES + LEAKAGE_SENSITIVE_FEATURES
NO_PAGEVALUE_FEATURES = BEHAVIORAL_FEATURES
