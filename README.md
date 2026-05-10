# When Machine Learning Fails: Online Shoppers

Research-oriented ML mini-project on the UCI Online Shoppers Purchasing Intention dataset. The goal is not benchmark chasing: the project intentionally forces a non-linear model to fail, diagnoses why it fails, and repairs the identified cause with a controlled intervention.

## Research Question

Does a non-linear classifier learn a spurious shortcut from calendar and acquisition proxies (`Month`, `TrafficType`, `VisitorType`, device/browser fields), such that it appears acceptable under an IID random split but loses robustness on late-year deployment months? If yes, can an invariant behavioral representation plus temporally adjacent validation improve out-of-period robustness without relying on that shortcut?

## Design

- Primary failure mode: shortcut learning under temporal distribution shift.
- Broken model: non-linear tree ensemble using shortcut-like variables.
- Controlled experiment: same model family and temporal test set, but manipulated feature representation.
- Correction: remove unstable shortcut proxies, use behavioral-intent features, validate on an adjacent temporal regime, and report uncertainty.
- Bonus failure mode: class imbalance and threshold dependence hiding purchase-class failure.

## Repository Structure

```text
ml-failure-online-shoppers/
|-- README.md
|-- requirements.txt
|-- data/
|   `-- online_shoppers_intention.csv
|-- notebooks/
|   |-- 01_broken_model.ipynb
|   |-- 02_failure_analysis.ipynb
|   `-- 03_corrected_model.ipynb
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- preprocessing.py
|   |-- training.py
|   |-- evaluation.py
|   |-- experiments.py
|   `-- visualization.py
|-- outputs/
|   |-- figures/
|   `-- metrics/
|-- report/
|   |-- final_report.md
|   `-- quantitative_claims.md
     -- Final_report.pdf
`-- docs/
    `-- assignment_summary.md
```

## How To Run

```bash
python -m venv .venv  
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order:

1. `notebooks/01_broken_model.ipynb`
2. `notebooks/02_failure_analysis.ipynb`
3. `notebooks/03_corrected_model.ipynb`

The notebooks use fixed random seeds and write final artifacts to:

- `outputs/figures/`
- `outputs/metrics/`

## Final Outputs

Important final tables:

- `table01_reference_vs_temporal.csv`
- `table04_controlled_causal_test.csv`
- `table05_temporal_drift_scores.csv`
- `table08_failure_evidence_bootstrap_ci.csv`
- `table09_before_after_bootstrap_deltas.csv`
- `table10_subgroup_month_before_after.csv`
- `table11_seed_variance_runs.csv`
- `table12_bonus_threshold_failure.csv`

Important final figures:

- `fig01a_confusion_random_broken.svg`
- `fig01b_confusion_temporal_broken.svg`
- `fig02_failure_evidence_bootstrap_ci.svg`
- `fig03_temporal_drift_scores.svg`
- `fig04a_permutation_importance_shortcut.svg`
- `fig04b_permutation_importance_behavioral.svg`
- `fig05_before_after_metrics.svg`
- `fig08_calibration_before_after.svg`
