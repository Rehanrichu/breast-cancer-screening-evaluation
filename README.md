# Breast Cancer Classification and Evaluation

I built this project to explore a question that matters as much as model accuracy: **how much should we trust the result of a machine-learning evaluation?**

The project uses scikit-learn's Wisconsin Diagnostic Breast Cancer dataset to classify tumours as malignant or benign from 30 numeric measurements of cell nuclei. I focused on a careful evaluation process: comparing several models, choosing a decision threshold using training data only, reporting uncertainty, and checking how results change when the train/test split changes.

> **Educational project only. This is not a medical device and must not be used to diagnose, screen, or guide treatment for anyone.**

## What the project does

1. Loads the 569-row, 30-feature benchmark included with scikit-learn.
2. Defines **malignant** as the positive class, so sensitivity represents the fraction of malignant cases identified by the model.
3. Creates a stratified train/test split before model selection and evaluation.
4. Compares a majority-class baseline with logistic regression, an RBF support-vector machine, random forest, and gradient boosting using repeated stratified cross-validation on the training data.
5. Keeps scaling inside model pipelines, reducing the risk of preprocessing leakage between folds.
6. Chooses a threshold from out-of-fold training predictions to target 98% sensitivity, then evaluates the selected model on the held-out test set.
7. Reports bootstrap confidence intervals, confusion matrices, calibration and permutation-importance plots, and results over 30 additional train/test splits.
8. Estimates how positive predictive value changes under lower assumed prevalence. These are calculations, not estimates of clinical performance.

## Main findings from the saved run

| Measure | Saved result |
|---|---:|
| Dataset size | 569 samples, 30 features |
| Selected model | Logistic regression |
| Training cross-validation ROC-AUC | 0.995 (mean) |
| Decision threshold chosen from training out-of-fold predictions | 0.207 |
| Accuracy on the first held-out split | 96.5% |
| Mean accuracy over 30 additional splits | 93.5% ± 2.4 percentage points |
| Mean sensitivity over 30 additional splits | 98.0% ± 2.2 percentage points |
| Mean specificity over 30 additional splits | 90.9% ± 4.2 percentage points |

The first split scored better than the average of the repeated splits, especially on specificity. That is one reason I do not treat a single test score as the complete story. The 30 test splits overlap and all come from the same dataset; they are **not** independent cohorts or external validation.

At the first split's tuned operating point (sensitivity 0.976, specificity 0.958), the saved report calculates positive predictive value (PPV) of about 93%, 55%, and 19% at assumed prevalences of 37%, 5%, and 1%. This illustrates why a high score on this case-enriched dataset should not be read as proof that the model would work as a population screening test.

## How to run it

Use Python 3.10 or newer. The saved results were generated with Python 3.12; small numerical differences can occur under other library versions.

```bash
python -m pip install -r requirements.txt
python screening.py
```

The full run compares models, creates plots, evaluates the held-out split, and repeats the procedure over 30 additional splits. It writes outputs into `results/`, replacing files there. To preserve the saved results, choose a different output directory:

```bash
python screening.py --outdir results-local
```

A quicker run for checking that the script works is:

```bash
python screening.py --robustness 2 --outdir results-smoke
python check.py
python -m unittest -v
```

`check.py` performs a smoke run in a temporary directory and checks core output properties. `test_screening.py` contains focused tests for the metric and threshold helpers. The GitHub Actions workflow runs these checks on pushes and pull requests.

## Repository map

```text
.
├── screening.py                 # End-to-end modelling and evaluation
├── check.py                     # Reproducible smoke checks
├── test_screening.py            # Unit tests for core helpers
├── requirements.txt             # Python dependencies
├── results/                     # Saved tables, metrics and figures
├── docs/
│   ├── DESIGN_NOTES.md           # Method choices, trade-offs and limitations
│   └── REFERENCES.md             # Dataset and library references
└── .github/workflows/checks.yml  # Automated checks on GitHub
```

See `results/README.md` for a guide to each saved output.

## Limitations and next steps

- The dataset is a well-known benchmark from one source, not a prospective screening population.
- The dataset's malignant-case prevalence is about 37%, much higher than in many general screening settings.
- Model choice and threshold selection use the training data; nested cross-validation could better account for selection uncertainty.
- The 30 repeated splits overlap and do not measure uncertainty on a new hospital or population.
- Calibration is only a limited visual check, and several features are correlated, which can dilute permutation importance.
- There is no external validation, demographic fairness analysis, prospective evaluation, or clinical review.

A sensible next step would be nested cross-validation, followed by external validation on an appropriately governed independent dataset. None of those steps would by itself make a model ready for clinical use.

## AI assistance

AI tools were used to assist with parts of the code and documentation. The project should be judged on the code, the reproducible checks, the results, and the author's ability to explain the decisions and limitations honestly.

## License

This project is released under the MIT License. See `LICENSE`.
