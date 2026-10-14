# Saved results

These files were included from the project's saved run. They let a reader inspect the reported outcome without immediately rerunning the full experiment.

| File | What it shows |
|---|---|
| `model_comparison_cv.csv` | Cross-validation mean and standard deviation for each model and metric. |
| `test_metrics.json` | Selected model, threshold, first split's test metrics, bootstrap intervals, feature summary and prevalence calculations. |
| `roc_out_of_fold.png` | ROC curves from out-of-fold training predictions. |
| `calibration.png` | A visual check of predicted probabilities versus observed fractions on training out-of-fold predictions. |
| `confusion_matrix_default_0.5.png` | Held-out confusion matrix using the default 0.5 threshold. |
| `confusion_matrix_tuned.png` | Held-out confusion matrix using the threshold selected for target sensitivity. |
| `permutation_importance.png` | Permutation importance measured by drop in test ROC-AUC. |
| `robustness_repeated_splits.csv` | Metrics from the additional train/test splits. |
| `robustness_repeated_splits.png` | Distribution of sensitivity and specificity across those splits. |

Running `screening.py` with its default output directory replaces these saved files. Use `--outdir results-local` to keep the included results unchanged.
