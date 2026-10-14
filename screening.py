"""
Project 1 - Breast Cancer Screening Classifier (careful evaluation)

Dataset : Wisconsin Diagnostic Breast Cancer (569 samples, 30 numeric features computed
          from digitised images of fine-needle-aspirate cell nuclei). Ships with scikit-learn.
Task    : Classify a tumour as malignant (positive class) or benign.
Goal    : Not "highest accuracy", but an evaluation a clinician-minded reviewer could trust:
            * the test set is split off FIRST and touched ONCE
            * all preprocessing lives inside Pipelines (no leakage)
            * models are compared with repeated stratified cross-validation vs. a trivial baseline
            * the decision threshold is chosen for a target SENSITIVITY (missing a cancer is the
              costly error), using out-of-fold predictions on the training data only
            * test metrics come with bootstrap confidence intervals, because n is small
            * the whole procedure is repeated over many different train/test splits, because one
              split of 114 test patients is itself a "lottery" (see Project 2)

EDUCATIONAL USE ONLY. This is not a medical device and must not be used for diagnosis.

Usage:
    python screening.py                       # default: target sensitivity 0.98
    python screening.py --target-sensitivity 0.95 --outdir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


# --------------------------------------------------------------------------- data
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return features and a label where 1 = malignant (the class we must not miss)."""
    bunch = load_breast_cancer(as_frame=True)
    # scikit-learn encodes malignant=0, benign=1. We flip it so that "positive" = malignant,
    # which makes recall (= sensitivity) mean "fraction of cancers we caught".
    assert list(bunch.target_names) == ["malignant", "benign"]
    y = (bunch.target == 0).astype(int)
    return bunch.data, y


def make_models(seed: int) -> dict:
    return {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed)),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=seed),
        "Gradient boosting": HistGradientBoostingClassifier(random_state=seed),
    }


# ------------------------------------------------------------------------ metrics
def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def pick_threshold(y_true: np.ndarray, p: np.ndarray, target_sensitivity: float) -> float:
    """Highest probability threshold whose sensitivity is still >= target (i.e. the most
    specific operating point that meets the sensitivity requirement)."""
    positives = (y_true == 1).sum()
    for t in np.unique(p)[::-1]:
        sens = ((p >= t) & (y_true == 1)).sum() / positives
        if sens >= target_sensitivity:
            return float(t)
    return float(p.min())


def bootstrap_ci(y_true, y_pred, p, n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile bootstrap 95% CIs on the test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = {"accuracy": [], "sensitivity": [], "specificity": [], "roc_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, pp = y_true[idx], y_pred[idx], p[idx]
        if yt.min() == yt.max():  # resample contained a single class; skip
            continue
        sens, spec = sensitivity_specificity(yt, yp)
        out["accuracy"].append(float((yt == yp).mean()))
        out["sensitivity"].append(sens)
        out["specificity"].append(spec)
        out["roc_auc"].append(float(roc_auc_score(yt, pp)))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def ppv_at_prevalence(sens: float, spec: float, prevalence: float) -> float:
    """Positive predictive value if the same test were used where only `prevalence` of people are
    malignant (Bayes' rule). Screening populations have far lower prevalence than this dataset."""
    tp = sens * prevalence
    fp = (1 - spec) * (1 - prevalence)
    return float(tp / (tp + fp))


def robustness_check(X, y, model_name: str, target_sensitivity: float, n_repeats: int, seed: int, cv_folds: int = 5) -> pd.DataFrame:
    """Repeat the complete procedure (split -> out-of-fold threshold -> single test evaluation)
    with different split seeds, to see how much the headline numbers depend on one lucky split."""
    rows = []
    for r in range(n_repeats):
        rs = seed + 1 + r
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
        oof = cross_val_predict(make_models(rs)[model_name], X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = pick_threshold(y_tr.to_numpy(), oof, target_sensitivity)
        fitted = make_models(rs)[model_name].fit(X_tr, y_tr)
        p = fitted.predict_proba(X_te)[:, 1]
        yt = y_te.to_numpy()
        pred = (p >= thr).astype(int)
        sens, spec = sensitivity_specificity(yt, pred)
        rows.append({"split_seed": rs, "threshold": thr, "sensitivity": sens, "specificity": spec,
                     "accuracy": float((pred == yt).mean()), "roc_auc": float(roc_auc_score(yt, p)),
                     "false_negatives": int(((pred == 0) & (yt == 1)).sum()),
                     "false_positives": int(((pred == 1) & (yt == 0)).sum())})
    return pd.DataFrame(rows)


def plot_robustness(rob: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].hist(rob.sensitivity, bins=10, color="#3b6ea5", edgecolor="white")
    axes[0].set(xlabel="Test sensitivity", ylabel="Number of splits",
                title=f"Sensitivity over {len(rob)} different splits")
    axes[1].hist(rob.specificity, bins=10, color="#e4572e", edgecolor="white")
    axes[1].set(xlabel="Test specificity", title=f"Specificity over {len(rob)} different splits")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------- the study
def compare_models(X_train, y_train, seed: int, n_splits: int = 5, n_repeats: int = 5) -> pd.DataFrame:
    """Repeated stratified cross-validation on the TRAINING data only."""
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "sensitivity": "recall",  # positive class = malignant
        "specificity": make_scorer(recall_score, pos_label=0),
        "precision": make_scorer(precision_score, zero_division=0),  # baseline predicts no positives
    }
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for name, model in make_models(seed).items():
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, error_score="raise")
        row = {"model": name}
        for m in scoring:
            vals = res[f"test_{m}"]
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_sd"] = vals.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def out_of_fold_probabilities(X_train, y_train, seed: int, n_splits: int = 5) -> dict[str, np.ndarray]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return {
        name: cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        for name, model in make_models(seed).items()
    }


def plot_roc(oof: dict, y_train, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in oof.items():
        fpr, tpr, _ = roc_curve(y_train, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_train, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="1 - specificity (false-positive rate)", ylabel="Sensitivity (true-positive rate)",
           title="Out-of-fold ROC curves (training data, 5-fold CV)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    labels = ["benign", "malignant"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="Actual", title=title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration(y_train, p, name: str, path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_train, p, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.set(xlabel="Mean predicted probability of malignancy", ylabel="Observed fraction malignant",
           title="Calibration (out-of-fold, training data)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_importance(names, means, sds, path: Path) -> None:
    order = np.argsort(means)[::-1][:10][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh([names[i] for i in order], means[order], xerr=sds[order], color="#3b6ea5")
    ax.set(xlabel="Drop in test ROC-AUC when feature is shuffled",
           title="Permutation importance (top 10, test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-sensitivity", type=float, default=0.98,
                    help="minimum fraction of malignant cases the chosen threshold must catch (CV estimate)")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="number of folds for cross-validation (default: 5)")
    ap.add_argument("--cv-repeats", type=int, default=5,
                    help="number of repeats for model comparison (default: 5)")
    ap.add_argument("--bootstrap-samples", type=int, default=2000,
                    help="number of bootstrap samples for confidence intervals (default: 2000)")
    ap.add_argument("--robustness", type=int, default=30,
                    help="repeat the whole split/threshold/test procedure over this many other splits (0 = skip)")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{int(y.sum())} malignant ({y.mean():.1%}), {int((1 - y).sum())} benign")

    # 1) split first; the test set is not used for anything until the very end
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)
    print(f"Train: {len(X_train)} rows | Test (held out): {len(X_test)} rows "
          f"({int(y_test.sum())} malignant)\n")

    # 2) model comparison on training data only
    if args.cv_folds < 2 or args.cv_repeats < 1 or args.bootstrap_samples < 1:
        ap.error("--cv-folds must be >= 2, --cv-repeats and --bootstrap-samples must be >= 1")

    table = compare_models(X_train, y_train, args.seed, n_splits=args.cv_folds, n_repeats=args.cv_repeats)
    table.round(4).to_csv(args.outdir / "model_comparison_cv.csv")
    show = table[[c for c in table.columns if c.endswith("_mean")]].round(3)
    print("Repeated 5x5 stratified CV on the training set (mean over 25 folds):")
    print(show.to_string(), "\n")

    candidates = table.drop(index="Majority-class baseline")
    best_name = candidates["roc_auc_mean"].idxmax()
    print(f"Selected model (highest CV ROC-AUC): {best_name}")

    # 3) choose the operating threshold from out-of-fold training predictions
    oof = out_of_fold_probabilities(X_train, y_train, args.seed, n_splits=args.cv_folds)
    plot_roc(oof, y_train, args.outdir / "roc_out_of_fold.png")
    plot_calibration(y_train, oof[best_name], best_name, args.outdir / "calibration.png")
    y_tr = y_train.to_numpy()
    threshold = pick_threshold(y_tr, oof[best_name], args.target_sensitivity)
    oof_sens, oof_spec = sensitivity_specificity(y_tr, (oof[best_name] >= threshold).astype(int))
    print(f"Threshold for sensitivity >= {args.target_sensitivity:.0%}: {threshold:.3f} "
          f"(out-of-fold sensitivity {oof_sens:.3f}, specificity {oof_spec:.3f})\n")

    # 4) fit on all training data, evaluate ONCE on the held-out test set
    model = make_models(args.seed)[best_name].fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    y_te = y_test.to_numpy()

    report = {"selected_model": best_name, "threshold": threshold,
              "target_sensitivity": args.target_sensitivity,
              "n_train": int(len(X_train)), "n_test": int(len(X_test)),
              "n_test_malignant": int(y_te.sum())}
    for label, thr in [("default_0.5", 0.5), ("tuned", threshold)]:
        pred = (p_test >= thr).astype(int)
        cm = confusion_matrix(y_te, pred, labels=[0, 1])
        sens, spec = sensitivity_specificity(y_te, pred)
        tn, fp, fn, tp = cm.ravel()
        ci = bootstrap_ci(y_te, pred, p_test, n_boot=args.bootstrap_samples, seed=args.seed)
        report[label] = {
            "threshold": thr, "confusion_matrix [[TN, FP], [FN, TP]]": cm.tolist(),
            "accuracy": float((pred == y_te).mean()), "sensitivity": sens, "specificity": spec,
            "precision_ppv": float(tp / (tp + fp)) if (tp + fp) else None,
            "npv": float(tn / (tn + fn)) if (tn + fn) else None,
            "roc_auc": float(roc_auc_score(y_te, p_test)),
            "pr_auc": float(average_precision_score(y_te, p_test)),
            "bootstrap_95ci": ci,
        }
        plot_confusion(cm, f"Test set, threshold {thr:.2f}", args.outdir / f"confusion_matrix_{label}.png")
        lo_s, hi_s = ci["sensitivity"]
        lo_p, hi_p = ci["specificity"]
        print(f"TEST @ threshold {thr:.3f} [{label}]: acc {report[label]['accuracy']:.3f} | "
              f"sens {sens:.3f} (95% CI {lo_s:.2f}-{hi_s:.2f}) | spec {spec:.3f} (95% CI {lo_p:.2f}-{hi_p:.2f}) | "
              f"FN={fn} FP={fp}")

    # 5) what is the model using?
    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=30,
                                 random_state=args.seed)
    plot_importance(list(X.columns), imp.importances_mean, imp.importances_std,
                    args.outdir / "permutation_importance.png")
    top = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(5)
    report["top_features_permutation"] = {k: round(float(v), 4) for k, v in top.items()}

    # 6) how much would these numbers move with a different split? and what about low prevalence?
    tuned = report["tuned"]
    report["ppv_if_prevalence_were"] = {
        f"{prev:.0%}": round(ppv_at_prevalence(tuned["sensitivity"], tuned["specificity"], prev), 3)
        for prev in (0.01, 0.05, float(y.mean()))}
    print("PPV of the tuned test if prevalence were 1% / 5% / dataset's "
          f"{y.mean():.0%}: " + " / ".join(f"{v:.0%}" for v in report["ppv_if_prevalence_were"].values()))
    if args.robustness > 0:
        rob = robustness_check(X, y, best_name, args.target_sensitivity, args.robustness, args.seed, cv_folds=args.cv_folds)
        rob.to_csv(args.outdir / "robustness_repeated_splits.csv", index=False)
        plot_robustness(rob, args.outdir / "robustness_repeated_splits.png")
        report["robustness_across_splits"] = {"n_splits": int(args.robustness)} | {
            m: {"mean": round(float(rob[m].mean()), 4), "sd": round(float(rob[m].std(ddof=1)), 4),
                "min": round(float(rob[m].min()), 4), "max": round(float(rob[m].max()), 4)}
            for m in ["sensitivity", "specificity", "accuracy", "roc_auc", "false_negatives", "false_positives"]}
        r = report["robustness_across_splits"]
        print(f"\nOver {args.robustness} other train/test splits (threshold re-chosen each time): "
              f"sensitivity {r['sensitivity']['mean']:.3f} +/- {r['sensitivity']['sd']:.3f} "
              f"(range {r['sensitivity']['min']:.2f}-{r['sensitivity']['max']:.2f}), "
              f"specificity {r['specificity']['mean']:.3f} +/- {r['specificity']['sd']:.3f}, "
              f"missed cancers per split {int(r['false_negatives']['min'])}-{int(r['false_negatives']['max'])}")

    (args.outdir / "test_metrics.json").write_text(json.dumps(report, indent=2))
    print("\nWith only "
          f"{int(y_te.sum())} malignant test cases the sensitivity CI is wide - treat the point estimate with caution.")
    print(f"Saved tables and figures to {args.outdir}/")


if __name__ == "__main__":
    main()
