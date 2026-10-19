# Breast Cancer Screening — careful evaluation of a classifier

One-sentence summary: a small, reproducible study that trains and evaluates classifiers on the scikit-learn Wisconsin Diagnostic Breast Cancer dataset with an emphasis on trustworthy evaluation (split-first design, no preprocessing leakage, thresholding for sensitivity, bootstrap confidence intervals, and robustness checks).

This is an educational project to demonstrate careful evaluation practices for binary classifiers in a screening-like setting. It is not a medical device and must not be used to diagnose, screen, or guide treatment for anyone.

---

## Quick start

Requirements: Python 3.10+ (the included run used Python 3.12). Install the runtime requirements:

```bash
python -m pip install -r requirements.txt
```

Run the full analysis (writes figures and JSON into `results/`):

```bash
python screening.py
```

Run and preserve the included example results (does not overwrite `results/`):

```bash
python screening.py --outdir results-local
```

Do a quick smoke check (fast, uses reduced CV/bootstrap settings):

```bash
python screening.py --robustness 2 --outdir results-smoke --cv-folds 3 --cv-repeats 1 --bootstrap-samples 100
python check.py               # runs the smoke check and basic assertions
python -m unittest -v         # unit tests for helpers
```

---

## What you’ll get

The script:
- Compares a trivial baseline with Logistic Regression, RBF SVM, Random Forest and HistGradientBoosting using repeated stratified cross-validation on the training set.
- Selects an operating probability threshold from out‑of‑fold training predictions to meet a target sensitivity (default 98%) and evaluates the chosen model once on a held-out test set.
- Reports bootstrap 95% confidence intervals, confusion matrices (default 0.5 and tuned threshold), calibration and permutation-importance plots.
- Repeats the entire procedure over multiple random train/test splits (`--robustness`) to show how results vary with the data split.

Saved example outputs (see `results/README.md`) include:
- model_comparison_cv.csv
- test_metrics.json
- roc_out_of_fold.png, calibration.png
- confusion_matrix_default_0.5.png, confusion_matrix_tuned.png
- permutation_importance.png
- robustness_repeated_splits.csv and PNG

---

## Main findings (from the included saved run)

- Dataset: 569 samples, 30 numeric features (scikit-learn builtin)
- Selected model (example run): Logistic regression
- Example tuned threshold: ~0.21
- Example held-out-test accuracy / sensitivity / specificity: shown in `results/test_metrics.json` (the included run reports high sensitivity—see saved results for exact numbers)
- Note: the saved run also computes PPV for assumed prevalences (1%, 5%, dataset prevalence) to show how predictive value changes in lower-prevalence settings.

---

## Why this design?

Key choices in screening.py:
- Split the test set first and touch it only once, after model selection and thresholding on training data.
- Keep all preprocessing inside sklearn Pipelines so cross-validation is not contaminated by preprocessing leakage.
- Choose thresholds from out‑of‑fold predictions to target sensitivity (missing cancers is a high-cost error).
- Report bootstrap confidence intervals because the held-out test set is small.
- Repeat the whole split/threshold/test flow across many random splits to reveal sensitivity to the train/test split.

For more design notes and trade-offs see: [docs/DESIGN_NOTES.md](https://github.com/Rehanrichu/breast-cancer-screening-evaluation/blob/main/docs/DESIGN_NOTES.md).

---

## How it’s organized

Top-level tree:

```
.
├── screening.py                 # End-to-end modelling and evaluation
├── check.py                     # Reproducible smoke checks
├── test_screening.py            # Unit tests for metric & threshold helpers
├── requirements.txt             # Python dependencies
├── results/                     # Saved tables, metrics and figures (example run)
├── docs/
│   ├── DESIGN_NOTES.md          # Method choices, trade-offs and limitations
│   └── REFERENCES.md            # Dataset and library references
└── .github/workflows/checks.yml # GitHub Actions for CI smoke checks
```

How it fits together: screening.py is the main script. It loads the scikit-learn dataset, does a single split-off test set, runs repeated CV on the training set to compare models, computes out-of-fold probabilities to choose a threshold that meets the target sensitivity, fits the chosen model on all training data and evaluates once on the held-out test set, and produces plots and a JSON report. `check.py` runs a fast end-to-end smoke test in a temp directory and asserts expected outputs.

---

## Reproducibility & testing

- Deterministic seeds are used (SEED = 42 by default) but randomness remains when you change seeds or increase the number of robustness repeats.
- Use `check.py` to run the smoke test. The full study uses larger numbers of CV repeats and bootstrap samples; the smoke checks run with smaller settings to keep CI time low.
- Unit tests: `python -m unittest -v` runs `test_screening.py` which validates metric helpers and thresholding logic.

---

## Limitations and responsible use

- This dataset is a single, well-known benchmark; it does not represent a prospective screening population.
- Prevalence in the dataset (~37% malignant) is much higher than typical screening contexts; PPV will be lower at real-world prevalences.
- The study demonstrates evaluation practices but does not substitute for external validation, demographic fairness analysis, clinical review, or prospective testing.
- Do not use models or thresholds produced here for patient care.

---

## Extending / experiments to try

- Use nested cross‑validation to better account for model/threshold selection uncertainty.
- Add external validation on an independent dataset under appropriate governance.
- Add demographic subgroup evaluation and fairness metrics.
- Replace permutation importance with SHAP analyses to get per-sample explanations.

---

## AI assistance

AI tools were used to assist with parts of the code and documentation; judge the project on the code, reproducible checks, results, and the author's explanations.

---

## License & citation

This project is released under the MIT License. See `LICENSE`. If you reuse the code or results in academic work, cite the repository and describe that the experiments were done on the scikit-learn Wisconsin Diagnostic Breast Cancer dataset.

---

## Contact

Maintainer: Rehanrichu (GitHub). Open issues or pull requests for feature requests, corrections, or to propose reproducibility improvements.
