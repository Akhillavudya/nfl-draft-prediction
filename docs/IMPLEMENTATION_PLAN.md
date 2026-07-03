# NFL Draft Prediction — Implementation Plan

**Competition:** GCI World 2026 / Omnicampus | **Deadline:** June 12 2026 11AM UTC  
**Metric:** ROC AUC (higher = better) | **Final rank = last submitted file**  
**Rules:** No external data, no hand-labeling, set seeds for reproducibility.

---

## Key Findings (from data audit — do not re-derive)

| Finding | Evidence |
|---|---|
| Missing `Age` → 1.8% draft rate vs 76.5% when present | Missingness flag alone = AUC 0.716 |
| All drill missingness is informative | 0 missing drills → 0.785 draft rate; 7 missing → 0.000 |
| `School` is strongly predictive | Draft rates 0.36–0.85; 97.6% test schools seen in train |
| Baseline kills both signals | Mean imputation + drops `School` entirely |
| Gradient boosters >> RF(depth=5) | Native NaN, deeper interactions, faster training |

---

## Implementation Phases (ordered by impact / time)

### Phase 0 — Beat the Baseline (~45–60 min)

Goal: single LightGBM model with missingness flags + School. Should significantly outperform baseline.

**Step 1: Load data**
```python
import numpy as np
import pandas as pd
from pathlib import Path

PATH = Path("input")
train = pd.read_csv(PATH / "train.csv")
test  = pd.read_csv(PATH / "test.csv")
```

**Step 2: Missingness flags (most important feature group)**
```python
drill_cols = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
              'Broad_Jump', 'Agility_3cone', 'Shuttle']

for df in [train, test]:
    for c in drill_cols:
        df[f'{c}_missing'] = df[c].isnull().astype(int)
    df['num_drills_missing'] = df[drill_cols].isnull().sum(axis=1)
    df['full_combine'] = (df['num_drills_missing'] == 0).astype(int)
```

**Step 3: BMI (keep from baseline)**
```python
for df in [train, test]:
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
```

**Step 4: School target encoding — MUST be fold-internal (avoid leakage)**
```python
# Do NOT compute this before the fold loop.
# Pattern: inside each fold, compute mean on train-fold rows only.
# For test, use full-train mean (computed after all CV).
# See the fold loop below for correct placement.
```

**Step 5: LightGBM 5-fold OOF loop**
```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
N_FOLDS = 5

cat_cols  = ['Player_Type', 'Position_Type', 'Position']  # label-encode or use lgb cat
drop_cols = ['Id', 'Drafted']

# Label-encode categoricals (simple, works for lgb)
from sklearn.preprocessing import LabelEncoder
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[c], test[c]]).astype(str)
    le.fit(combined)
    train[c] = le.transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))

feature_cols = [c for c in train.columns if c not in drop_cols + ['School']]

X = train[feature_cols].copy()
y = train['Drafted'].values
X_test = test[feature_cols].copy()

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))
auc_scores = []

# Full-train school encoding (for test predictions)
school_global = train.groupby('School')['Drafted'].mean()

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    # --- FOLD-INTERNAL school target encoding ---
    school_map = train.iloc[tr_idx].groupby('School')['Drafted'].mean()
    global_mean = y_tr.mean()
    smooth = 10  # smoothing factor

    def smooth_encode(schools, mapping, global_mean, smooth=10):
        counts = train.groupby('School')['School'].count()
        enc = []
        for s in schools:
            if s in mapping.index:
                n = counts.get(s, 0)
                enc.append((n * mapping[s] + smooth * global_mean) / (n + smooth))
            else:
                enc.append(global_mean)
        return np.array(enc)

    X_tr['School_enc'] = smooth_encode(train.iloc[tr_idx]['School'], school_map, global_mean)
    X_va['School_enc'] = smooth_encode(train.iloc[va_idx]['School'], school_map, global_mean)

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        verbose=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )

    oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, oof_preds[va_idx])
    auc_scores.append(auc)
    print(f"Fold {fold+1}  AUC: {auc:.4f}")

    # Test predictions (add school encoding for test)
    X_te_fold = X_test.copy()
    X_te_fold['School_enc'] = smooth_encode(test['School'], school_map, global_mean)
    test_preds += model.predict_proba(X_te_fold)[:, 1] / N_FOLDS

print(f"\nOOF AUC: {roc_auc_score(y, oof_preds):.4f}  |  Mean fold AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
```

**Step 6: Save submission**
```python
sub = pd.read_csv(PATH / "sample_submission.csv")
sub["Drafted"] = test_preds
sub.to_csv("submission_lgbm_phase0.csv", index=False)
print("Saved.")
```

---

### Phase 1 — Core Feature Engineering (~1–2 hr)

Add these features to both `train` and `test` **before** the fold loop.

**A. Physical composites**
```python
for df in [train, test]:
    df['power']        = df['Vertical_Jump'] * df['Weight']
    df['speed_score']  = df['Weight'] / (df['Sprint_40yd'] ** 4 + 1e-9)
    df['agility_diff'] = df['Agility_3cone'] - df['Shuttle']
    df['jump_ratio']   = df['Broad_Jump'] / (df['Vertical_Jump'] + 1e-9)
    df['bench_per_kg'] = df['Bench_Press_Reps'] / (df['Weight'] + 1e-9)
```

**B. Position-relative drill z-scores (fold-internal for strictness, or full-train is low-risk)**
```python
# Simpler version: compute on full train (position stats don't depend on target)
drill_vals = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
              'Broad_Jump', 'Agility_3cone', 'Shuttle']

pos_stats = train.groupby('Position')[drill_vals].agg(['mean', 'std'])

def add_position_z(df, stats):
    for d in drill_vals:
        mu  = df['Position'].map(stats[(d, 'mean')])
        std = df['Position'].map(stats[(d, 'std')]).fillna(1).replace(0, 1)
        df[f'{d}_z_pos'] = (df[d] - mu) / std
    return df

train = add_position_z(train, pos_stats)
test  = add_position_z(test,  pos_stats)
```

**C. Frequency encodings**
```python
school_freq = pd.concat([train['School'], test['School']]).value_counts()
for df in [train, test]:
    df['School_freq'] = df['School'].map(school_freq).fillna(0)

pos_freq = train['Position'].value_counts()
for df in [train, test]:
    df['Position_freq'] = df['Position'].map(pos_freq).fillna(0)
```

After adding these, re-run the Phase 0 fold loop and compare OOF AUC.

---

### Phase 2 — Tune & Ensemble (~1–2 hr)

**A. Optuna hyperparameter search (run ~30–50 trials)**
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = dict(
        n_estimators     = 1000,
        learning_rate    = trial.suggest_float("lr", 0.01, 0.1, log=True),
        num_leaves       = trial.suggest_int("num_leaves", 15, 255),
        max_depth        = trial.suggest_int("max_depth", 3, 12),
        min_child_samples= trial.suggest_int("min_child", 5, 100),
        feature_fraction = trial.suggest_float("feat_frac", 0.5, 1.0),
        bagging_fraction = trial.suggest_float("bag_frac", 0.5, 1.0),
        bagging_freq     = 1,
        reg_alpha        = trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        reg_lambda       = trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        random_state     = SEED, verbose=-1
    )
    # Run same 5-fold OOF loop with these params; return mean OOF AUC
    # ... (copy the fold loop, replace model params, return roc_auc_score(y, oof))
    pass

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Best OOF AUC:", study.best_value)
print("Best params:", study.best_params)
```

**B. CatBoost model (raw School as cat_feature)**
```python
from catboost import CatBoostClassifier

# Use School as a raw string — CatBoost handles it natively
cat_features_cb = ['School', 'Player_Type', 'Position_Type', 'Position']

# Same fold loop, but pass cat_features to the model
model_cb = CatBoostClassifier(
    iterations=1000, learning_rate=0.05, depth=6,
    random_seed=SEED, verbose=0, eval_metric='AUC',
    early_stopping_rounds=50
)
```

**C. Rank-average ensemble**
```python
from scipy.stats import rankdata

def rank_avg(preds_list):
    n = len(preds_list[0])
    ranked = [rankdata(p) / n for p in preds_list]
    return np.mean(ranked, axis=0)

# Blend LGBM + CatBoost (+ XGBoost if added)
ensemble_preds = rank_avg([lgbm_test_preds, catboost_test_preds])

# Pick blend weights by OOF AUC
# w_lgbm * lgbm_oof + w_cb * cb_oof, grid over w in [0,1]
best_w, best_auc = None, 0
for w in np.arange(0.0, 1.05, 0.05):
    blended_oof = w * lgbm_oof + (1 - w) * catboost_oof
    a = roc_auc_score(y, blended_oof)
    if a > best_auc:
        best_auc, best_w = a, w
print(f"Best blend weight (LGBM): {best_w:.2f}  OOF AUC: {best_auc:.4f}")
```

---

### Phase 3 — Stabilize & Final Submission (~last hour)

**Seed averaging (reduces variance on 696-row test)**
```python
SEEDS = [42, 123, 2025]

all_test_preds = []
all_oof_preds  = []

for seed in SEEDS:
    # re-run the complete fold loop with random_state=seed
    # append oof_preds, test_preds to lists
    pass

final_oof  = np.mean(all_oof_preds,  axis=0)
final_test = np.mean(all_test_preds, axis=0)

print(f"Final OOF AUC (seed-averaged): {roc_auc_score(y, final_oof):.4f}")
```

**Final submission**
```python
sub = pd.read_csv(PATH / "sample_submission.csv")
sub["Drafted"] = final_test
sub.to_csv("submission_FINAL.csv", index=False)

# Sanity checks
assert len(sub) == 696
assert sub["Drafted"].between(0, 1).all()
assert list(sub.columns) == ["Id", "Drafted"]
print("Ready to submit.")
```

---

## Critical Rules & Traps

| Rule | Why |
|---|---|
| **Target encoding MUST be fold-internal** | Computing School mean on full train before CV = data leakage → inflated CV, disappointing LB |
| **Trust OOF AUC, not Public LB** | Public LB is a small noisy subset of 696 test rows |
| **Final rank = last submitted file** | Submit your best-CV model last |
| **No `if Age_missing → 0` rules** | Prohibited as hand-labeling; use `Age_missing` as a model *feature* instead |
| **Set all seeds** | Reproducibility required; top finishers submit runnable Colab code |
| **No external data** | Prohibited |

---

## Results Log (fill this in as you go)

| Experiment | Key changes | OOF AUC | Public LB |
|---|---|---|---|
| 0. Baseline | RF depth=5, mean impute, no School | — | — |
| 1. Phase 0 | LGBM + miss flags + School enc | | |
| 2. Phase 1 | + composites + pos z-scores + freq enc | | |
| 3. Phase 2 | Optuna tuned LGBM | | |
| 4. Phase 2b | + CatBoost blend | | |
| 5. Phase 3 | + seed averaging | | |

**Select your last submission = row with highest OOF AUC.**

---

## Expected OOF AUC Trajectory

- Baseline (RF depth=5): ~0.75–0.78 (estimate)
- Phase 0 (LGBM + miss flags + School): ~0.83–0.87
- Phase 1 (+ position features + composites): ~0.85–0.89
- Phase 2 (tuned + ensemble): ~0.87–0.91
- Phase 3 (seed-averaged): marginal stabilization +0.003–0.01

> Values are estimates from data analysis — verify against your actual CV numbers.
