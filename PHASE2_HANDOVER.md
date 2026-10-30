# Phase 2 – Random Forest Baseline Hand‑over Document

## Objective
Develop a traditional machine‑learning baseline for the Crypto‑AML task using a **Random Forest** classifier.

## Starting point & Phase 1 dependency
- Repository: **Phase 1** – https://github.com/bhanuprakashpindi-creator/crypto-aml-gnn-phase1
- All preprocessing, graph construction and feature generation are taken **as‑is** from Phase 1.

## Phase 1 processed inputs used
- `data/processed/elliptic_x_reduced.npy` – reduced feature matrix (shape [203 769 × 145])
- `data/processed/elliptic_y.npy` – integer labels (0 = licit, 1 = illicit, -1 = unknown)
- `data/processed/elliptic_masks.npz` – boolean masks for train/validation/test timesteps
- `data/processed/elliptic_idx_to_tx.json` – mapping from matrix index → original transaction ID

## Label encoding & handling of unknowns
- **1 = illicit**
- **0 = licit**
- **-1 = unknown** (these rows are **excluded** from all supervised training and evaluation steps).

## Temporal split (as defined in Phase 1)
| Split | Timesteps | Labeled nodes | Illicit | Licit |
|-------|-----------|---------------|---------|-------|
| **Train** | 1‑34 | 29 894 | 3 462 | 26 432 |
| **Validation** | 35‑41 | 7 829 | 675 | 7 154 |
| **Test** | 42‑49 | 8 841 | 408 | 8 433 |

## Feature handling
- The script **never** uses transaction IDs as model features; they are only loaded to map predictions back to the original IDs.
- No additional preprocessing (scaling, PCA, etc.) is fitted on validation or test data – the raw reduced features are used directly.

## Random Forest configuration
```
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

## Model selection / tuning
- **No test‑set tuning** or hyper‑parameter search is performed. The configuration above is fixed for the baseline.
- Validation metrics are computed only for reporting; they do not influence the final model.

## Final evaluation (test set)
| Metric | Value |
|--------|-------|
| **Precision** | 0.781377 |
| **Recall** | 0.473039 |
| **F1‑score** | 0.589313 |
| **PR‑AUC** | 0.551840 |

## Confusion matrix (test set)
|            | Predicted Illicit | Predicted Licit |
|------------|-------------------|-----------------|
| **Actual Illicit** | **TP = 193** | **FN = 215** |
| **Actual Licit**   | **FP = 54**  | **TN = 8 379** |

## Output artefacts (all written under the repository root)
- `models/random_forest.joblib` – serialized model
- `results/tables/random_forest_metrics.csv` – CSV with the four test metrics
- `results/random_forest_confusion_matrix.csv` – confusion‑matrix CSV
- `results/random_forest_predictions.csv` – `txId,prediction,prob_illicit` for every labeled test transaction (8 841 rows)
- `results/random_forest_manifest.json` – seed, hyper‑parameters, input file paths, split counts, etc.

## Reproducible training command
```bash
python -m src.models.random_forest
```
Running the command from the repository root reproduces the exact results above (seed = 42).

## Test suite result
All Phase 2 unit tests pass:
```
6 tests, 0 failures
```

## Data‑leakage / audit outcome
All **A‑J** checks were performed and **PASS**:
- Only labeled (0/1) samples used.
- No unknown labels or transaction IDs as features.
- No test‑set information leaked into training.
- Feature matrix untouched.
- No preprocessing fitted on validation/test.
- No test‑set tuning.

## Limitations
- The baseline does **not** incorporate graph‑structured information; it relies solely on the reduced hand‑crafted features.
- Class‑weight handling mitigates imbalance but may not be optimal for the highly skewed illicit class.
- No hyper‑parameter search; results are a lower bound for more sophisticated models.

## Instructions for the Phase 3 teammate
1. Create a new repository for Phase 3 (e.g., `crypto-aml-gnn-phase3`).
2. Clone this Phase 2 repository as the starting point and preserve all existing Phase 1 + Phase 2 source code, tests, documentation, and results.
3. Note that `data/raw/` and `data/processed/` are intentionally excluded from GitHub because they contain large dataset/processed artifacts. Obtain the required Phase 1 processed artifacts through the team's agreed Phase 1 data handoff or regenerate them using the Phase 1 pipeline.
4. Use the same feature definitions, transaction‑ID/node‑index mapping, label encoding, and temporal train/validation/test split established in Phase 1 and Phase 2.
5. For GCN, GraphSAGE, and GAT experiments, exclude unknown labels (-1) from supervised training/evaluation and do not use test results for model selection.
6. Preserve the Phase 2 Random Forest baseline so Phase 3 and Phase 4 can compare GNN results against it.
7. Document all new preprocessing steps, hyperparameters, checkpoints, metrics, and results in a Phase 3 handover document.
8. Create a NEW GitHub repository for Phase 3 and do not push Phase 3 changes directly to the Phase 2 repository.

---
*This hand‑over document was generated automatically to capture the complete state of Phase 2.*
