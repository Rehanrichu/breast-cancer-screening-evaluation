# Design notes

## Project question

The main goal is not to maximise a single accuracy score. It is to show how modelling decisions and test-split variability affect the conclusions drawn from a small benchmark dataset.

## Data and labels

The project uses `sklearn.datasets.load_breast_cancer`, which provides 569 records and 30 numeric measurements. Scikit-learn's original target encoding has benign as class 1 and malignant as class 0. `load_data()` reverses that encoding so malignant is the positive class. This keeps sensitivity/recall aligned with the stated task of identifying malignant cases.

## Evaluation flow

1. Make a stratified 80/20 train/test split.
2. Compare the majority-class baseline and four classifiers on the training portion using repeated stratified 5-fold cross-validation (five repeats).
3. Select the non-baseline model with the highest mean cross-validated ROC-AUC.
4. Generate out-of-fold training predictions and choose the highest probability threshold that meets the configured target sensitivity.
5. Fit the selected model on the complete training portion and evaluate the held-out test portion.
6. Report confusion matrices, several metrics and bootstrap percentile intervals; calculate PPV for different assumed prevalence values.
7. Repeat split, threshold selection and evaluation for additional seeds to illustrate how metrics vary across splits.

## Important implementation choices

- **Pipeline scaling:** scaling is inside pipelines for logistic regression and the RBF SVM, so each cross-validation fold learns its scaling values from its own training fold.
- **Baseline:** the majority-class classifier makes it possible to compare model performance with a simple no-skill reference.
- **Threshold selection:** the test set is not used to select the threshold. Out-of-fold training predictions are used instead.
- **Metrics:** sensitivity and specificity accompany accuracy because different error types matter differently in a cancer-screening context.
- **Reproducibility:** random seeds are set for splitting, models, bootstrapping and repeated-split experiments.

## How to interpret the results

The saved run's first held-out split had 96.5% accuracy. Across 30 additional splits, mean accuracy was 93.5% with a standard deviation of about 2.4 percentage points; mean sensitivity was 98.0%, and mean specificity was 90.9%. The first split therefore gave a more favourable result than the average of the repeated runs, particularly for specificity.

These repeated splits reuse the same 569 records and overlap. Their spread is descriptive of split-to-split variation on this dataset; it is not external validation and should not be described as a confidence interval for performance at a new hospital.

The first-split tuned sensitivity and specificity imply lower PPV when assumed prevalence is lowered. That calculation depends on the selected sensitivity/specificity carrying over to the target setting, which has not been demonstrated here.

## Limitations

- Single benchmark dataset from one source; no external validation.
- Dataset class prevalence is far higher than in many screening populations.
- Model selection and threshold selection share the training data, so selection uncertainty is not fully accounted for. Nested cross-validation is a possible improvement.
- The calibration plot is sparse and is not enough to establish reliable probabilities.
- Correlated features can make individual permutation-importance values look small.
- No demographic fields are available for a meaningful fairness assessment.
- The work is educational and does not support clinical use.
