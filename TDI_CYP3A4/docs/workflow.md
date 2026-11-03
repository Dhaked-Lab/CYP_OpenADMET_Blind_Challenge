# Workflow

```
raw CSVs ─> data_audit ─> data_clean ─> features ─> similarity ─┐
                              │            │                    │
                              ├─> scaffold_cv ──────────────────┤
                              └─> embeddings ──┐                │
                                               v                v
      oof_base (features, similarity, folds) ──┐
      oof_emb  (+ embeddings) ─────────────────┤
      oof_tab  (folds, embeddings; fold PCA) ──┼─> ensemble_selection ─> threshold_selection
      oof_tabcp(folds, embeddings; fold PCA) ──┘                                  │
                                                                                  v
 model_training ─┐                                                           (fraction)
 blind_features ─┼─> blind_embeddings ─> prediction ───────────────────────> submission
 ensemble weights┘
```

Each arrow is a declared dependency in `src/cyp_tdi/registry.py`; `python run_pipeline.py --list-steps` prints the table.
A step is skipped when its fingerprint (implementation version, relevant config, input file hashes, upstream output hashes) is unchanged.

Out-of-fold members are produced per scaffold fold (K=5, seed 7). Ensemble weights are chosen per outer fold on the other folds' OOF predictions
(nested greedy, 12 rounds). The final models use all training rows; TabICL is refit on the full training set at prediction time. No blind
label or leaderboard information enters any selection.
