# Elliptic Dataset Statistics and Pipeline Manifest

- **Execution Date**: 2026-10-30 22:20:19
- **Total Transaction Nodes**: 203,769
- **Total Directed Edges**: 234,355
- **Raw Features**: 165 numeric features (93 local, 72 aggregated)
- **Reduced Features (Paper Best Configuration)**: 143 numeric features (22 collinear aggregated features pruned on train split)

## Class Distribution
- **Illicit (1)**: 4,545 (2.23%)
- **Licit (0)**: 42,019 (20.62%)
- **Unknown (-1)**: 157,205 (77.15%)
- **Total Labeled Subgraph**: 46,564 transactions (9.76% illicit class ratio)

## Temporal Split Breakdown
- **Train (Timesteps 1..34)**:
  - Total Nodes: 136,265
  - Labeled Nodes: 29,894 (3,462 illicit, 26,432 licit)
  - Illicit Ratio (Labeled): 11.58%
- **Validation (Timesteps 35..41)**:
  - Total Nodes: 30,680
  - Labeled Nodes: 7,829 (675 illicit, 7,154 licit)
  - Illicit Ratio (Labeled): 8.62%
- **Test (Timesteps 42..49)**:
  - Total Nodes: 36,824
  - Labeled Nodes: 8,841 (408 illicit, 8,433 licit)
  - Illicit Ratio (Labeled): 4.61%

## Leakage Audit
- Cross-split edges: **0**
- Topological Leakage Status: **No edges cross temporal split boundaries. The dataset consists of completely isolated temporal graph snapshots.**
