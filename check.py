"""Quick end-to-end smoke check; does not overwrite the included saved results."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "results"
        cmd = [
            sys.executable, str(ROOT / "screening.py"),
            "--cv-folds", "3", "--cv-repeats", "1", "--bootstrap-samples", "100",
            "--robustness", "2", "--outdir", str(out),
        ]
        p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if p.returncode:
            print(p.stdout[-3000:])
            print(p.stderr[-3000:])
            return 1
        metrics_path = out / "test_metrics.json"
        if not metrics_path.exists():
            print("FAIL: test_metrics.json was not created")
            return 1
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert m["n_train"] + m["n_test"] == 569
        assert m["selected_model"] in {"Logistic regression", "SVM (RBF)", "Random forest", "Gradient boosting"}
        assert 0 < m["threshold"] < 1
        default_cm = m["default_0.5"]["confusion_matrix [[TN, FP], [FN, TP]]"]
        tuned_cm = m["tuned"]["confusion_matrix [[TN, FP], [FN, TP]]"]
        assert sum(map(sum, default_cm)) == m["n_test"]
        assert sum(map(sum, tuned_cm)) == m["n_test"]
        assert m["tuned"]["sensitivity"] >= m["default_0.5"]["sensitivity"]
        expected_figures = [
            "roc_out_of_fold.png", "calibration.png", "confusion_matrix_tuned.png",
            "confusion_matrix_default_0.5.png", "permutation_importance.png",
            "robustness_repeated_splits.png",
        ]
        for name in expected_figures:
            assert (out / name).exists(), f"Missing figure: {name}"
    print("PASS: breast-cancer evaluation smoke checks")
    print("Note: this uses reduced CV/bootstrap settings; use the default command for the full study.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
