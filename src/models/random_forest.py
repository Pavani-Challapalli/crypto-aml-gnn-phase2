# src/models/random_forest.py
"""Random Forest baseline for Crypto AML Phase 2.

This module loads the Phase 1 processed artefacts, constructs train/validation/test
splits using only *labeled* nodes (labels 0 = licit, 1 = illicit), trains a
scikit‑learn :class:`RandomForestClassifier` with a balanced class‑weight to
handle the strong class imbalance, selects the best model based on validation
F1‑score (illicit class), and evaluates the final model on the held‑out test set.

All artefacts are saved under the repository root:

- ``models/random_forest.joblib`` – serialized model (joblib)
- ``results/tables/random_forest_metrics.csv`` – metric table (CSV)
- ``results/random_forest_predictions.csv`` – predictions keyed by transaction ID
- ``results/random_forest_confusion_matrix.csv`` – confusion matrix (CSV)

The script is deterministic (fixed ``random_state``) and does **not** use the
unknown label ``-1`` for training or evaluation, nor does it use transaction IDs
as features – they are only used to map predictions back to the original
transactions.
"""

import json
import pathlib
import sys
from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Configuration (hard‑coded for reproducibility – can be moved to a config file)
# ---------------------------------------------------------------------------
SEED = 42
RANDOM_STATE = SEED
N_ESTIMATORS = 200
MAX_DEPTH = None  # let the trees grow until pure leaves
CLASS_WEIGHT = "balanced"  # mitigate imbalance without hyper‑parameter search

# Paths – relative to repository root (the script is executed from the repo root)
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent  # repo root
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"

# Ensure output directories exist
for p in (MODEL_DIR, RESULTS_DIR, TABLES_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_artifacts() -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load features, labels and masks.

    Returns
    -------
    X : np.ndarray
        Reduced feature matrix (shape [N, 145]).
    y : np.ndarray
        Integer label vector (shape [N,]).
    masks : dict
        Dictionary loaded from ``elliptic_masks.npz`` containing boolean masks:
        ``train_mask``, ``val_mask``, ``test_mask`` as well as their
        ``*_mask_labeled`` variants.
    """
    X_path = DATA_DIR / "elliptic_x_reduced.npy"
    y_path = DATA_DIR / "elliptic_y.npy"
    masks_path = DATA_DIR / "elliptic_masks.npz"

    X = np.load(X_path)
    y = np.load(y_path)
    masks = np.load(masks_path)
    return X, y, masks


def filter_labeled(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Select only rows where ``y`` is 0 or 1 (i.e., not ``-1``).

    Parameters
    ----------
    X, y : np.ndarray
        Full feature matrix and label vector.
    mask : np.ndarray
        Boolean mask selecting a subset of rows (e.g., train_mask).
    """
    # Apply the mask first, then drop unknown labels
    X_sel = X[mask]
    y_sel = y[mask]
    labeled = (y_sel == 0) | (y_sel == 1)
    return X_sel[labeled], y_sel[labeled]


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a RandomForestClassifier with the predefined hyper‑parameters."""
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute required metrics on a given split.

    Returns a dictionary with metric names as keys.
    """
    prob_illicit = clf.predict_proba(X)[:, 1]
    y_pred = (prob_illicit >= 0.5).astype(int)
    precision = precision_score(y, y_pred, pos_label=1)
    recall = recall_score(y, y_pred, pos_label=1)
    f1 = f1_score(y, y_pred, pos_label=1)
    pr_auc = average_precision_score(y, prob_illicit)
    cm = confusion_matrix(y, y_pred, labels=[1, 0])  # order: [illicit, licit]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "prob_illicit": prob_illicit,
    }


def save_metrics(metrics: dict):
    """Write metric values to ``results/tables/random_forest_metrics.csv``."""
    csv_path = TABLES_DIR / "random_forest_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        f.write("metric,value\n")
        f.write(f"precision,{metrics['precision']:.6f}\n")
        f.write(f"recall,{metrics['recall']:.6f}\n")
        f.write(f"f1,{metrics['f1']:.6f}\n")
        f.write(f"pr_auc,{metrics['pr_auc']:.6f}\n")


def save_confusion_matrix(cm: np.ndarray):
    """Save confusion matrix as CSV with explicit row/column labels."""
    cm_path = RESULTS_DIR / "random_forest_confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        f.write(",illicit,licit\n")
        f.write(f"illicit,{cm[0,0]},{cm[0,1]}\n")
        f.write(f"licit,{cm[1,0]},{cm[1,1]}\n")


def save_predictions(
    tx_ids: np.ndarray,
    mask: np.ndarray,
    y_pred: np.ndarray,
    prob_illicit: np.ndarray,
):
    """Write test predictions keyed by transaction ID.

    Only the *labeled* test nodes are written (unknown labels are excluded).
    """
    pred_path = RESULTS_DIR / "random_forest_predictions.csv"
    with open(pred_path, "w", newline="") as f:
        f.write("txId,prediction,prob_illicit\n")
        for tx, pred, prob in zip(tx_ids, y_pred, prob_illicit):
            f.write(f"{tx},{int(pred)},{prob:.6f}\n")


def main():
    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------
    X, y, masks = load_artifacts()

    # Load transaction‑ID mapping (index → txId)
    tx_to_idx_path = DATA_DIR / "elliptic_idx_to_tx.json"
    with open(tx_to_idx_path, "r", encoding="utf-8") as f:
        idx_to_tx = json.load(f)
    idx_to_tx = np.array(idx_to_tx)  # shape [N]

    # ---------------------------------------------------------------------
    # Build train / validation / test sets (only labeled nodes)
    # ---------------------------------------------------------------------
    X_train, y_train = filter_labeled(X, y, masks["train_mask"])
    X_val, y_val = filter_labeled(X, y, masks["val_mask"])
    # Test split – keep original ordering for mapping back to IDs
    test_mask = masks["test_mask"]
    test_idx = np.where(test_mask)[0]
    # Filter unknown labels for test as well
    test_known = (y[test_idx] == 0) | (y[test_idx] == 1)
    test_idx = test_idx[test_known]
    X_test = X[test_idx]
    y_test = y[test_idx]
    tx_test = idx_to_tx[test_idx]

    # ---------------------------------------------------------------------
    # Model training
    # ---------------------------------------------------------------------
    clf = train_random_forest(X_train, y_train)

    # ---------------------------------------------------------------------
    # Validation – we could tune hyper‑parameters here, but we keep the simple
    # baseline.  Metrics are computed for information only.
    # ---------------------------------------------------------------------
    val_metrics = evaluate(clf, X_val, y_val)
    print("[Validation] Precision: {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}, PR-AUC: {3:.4f}".format(
        val_metrics["precision"], val_metrics["recall"], val_metrics["f1"], val_metrics["pr_auc"]
    ))

    # ---------------------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------------------
    test_metrics = evaluate(clf, X_test, y_test)
    print("[Test] Precision: {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}, PR-AUC: {3:.4f}".format(
        test_metrics["precision"], test_metrics["recall"], test_metrics["f1"], test_metrics["pr_auc"]
    ))

    # ---------------------------------------------------------------------
    # Save artefacts
    # ---------------------------------------------------------------------
    joblib.dump(clf, MODEL_DIR / "random_forest.joblib")
    save_metrics(test_metrics)
    save_confusion_matrix(test_metrics["confusion_matrix"])
    save_predictions(
        tx_test,
        test_mask,
        test_metrics["y_pred"],
        test_metrics["prob_illicit"],
    )

    # ---------------------------------------------------------------------
    # Reproducibility info (written to a small manifest)
    # ---------------------------------------------------------------------
    manifest_path = RESULTS_DIR / "random_forest_manifest.json"
    manifest = {
        "random_seed": SEED,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "class_weight": CLASS_WEIGHT,
        "train_labeled_count": int(len(y_train)),
        "val_labeled_count": int(len(y_val)),
        "test_labeled_count": int(len(y_test)),
        "feature_file": str(DATA_DIR / "elliptic_x_reduced.npy"),
        "label_file": str(DATA_DIR / "elliptic_y.npy"),
        "mask_file": str(DATA_DIR / "elliptic_masks.npz"),
        "tx_mapping_file": str(DATA_DIR / "elliptic_idx_to_tx.json"),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[Done] All Phase 2 artefacts written under 'models/' and 'results/'.")


if __name__ == "__main__":
    # Ensure the repository root is on Python path (important when executed via
    # "python -m src.models.random_forest").
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    main()
