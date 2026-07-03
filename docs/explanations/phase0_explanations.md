# Phase 0 — Code Explanation (Beginner Friendly)

This document explains every block of `phase0_lgbm.py` in plain language.
No prior machine learning experience is assumed.

---

## Block 1 — Importing Libraries

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
```

### What it does
Brings in external tools (called *libraries*) that we need throughout the script.

### Why each library
| Library | Purpose |
|---|---|
| `numpy` | Fast math on arrays of numbers (e.g. averages, zeros, array operations) |
| `pandas` | Reads CSV files and lets us work with tables (called *DataFrames*) |
| `lightgbm` | The machine learning model we train — a fast, accurate decision-tree booster |
| `pathlib.Path` | A clean way to build file paths that works on Windows and Mac |
| `StratifiedKFold` | Splits our data into folds while keeping the class ratio balanced (more on this later) |
| `roc_auc_score` | The competition metric — measures how well the model ranks drafted vs undrafted players |
| `LabelEncoder` | Converts text categories (like "WR", "QB") into numbers so the model can read them |

---

## Block 2 — Loading the Data

```python
PATH = Path("input")
train = pd.read_csv(PATH / "train.csv")
test  = pd.read_csv(PATH / "test.csv")
```

### What it does
- `PATH` points to the `input/` folder where our CSV files live.
- `train` is a table with 2 781 players — we *know* whether each was drafted (`Drafted = 1`) or not (`Drafted = 0`). We use this to teach the model.
- `test` is a table with 696 players — we do *not* know whether they were drafted. The model has to predict this for us.

### Why
Machine learning always needs two datasets:
- **Training data** — examples with known answers the model learns from.
- **Test data** — new examples we want the model to make predictions on.

---

## Block 3 — Missingness Flags (Most Important Feature Group)

```python
drill_cols = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
              'Broad_Jump', 'Agility_3cone', 'Shuttle']

for df in [train, test]:
    for c in drill_cols:
        df[f'{c}_missing'] = df[c].isnull().astype(int)
    df['num_drills_missing'] = df[drill_cols].isnull().sum(axis=1)
    df['full_combine']       = (df['num_drills_missing'] == 0).astype(int)
```

### What it does
For every combine drill column (and Age), we create a new column that is:
- `1` if the value is **missing** (blank/NaN)
- `0` if the value is **present**

We also create:
- `num_drills_missing` — how many of the 7 columns are missing for each player
- `full_combine` — `1` if *all* drills are present, `0` otherwise

### Why this is the single most valuable thing we do
When a player does **not** show up to the NFL Combine or skips drills, it usually means they were not invited — and only projected draft picks get invited.

From the data audit:
- Players with `Age` **missing** → draft rate **1.8%** (almost never drafted)
- Players with `Age` **present** → draft rate **76.5%** (almost always drafted)

The *absence* of data is itself a powerful signal. The baseline model treated missing values as "unknown" and filled them with the column average, throwing away this signal entirely. We preserve it by turning the absence into an explicit 0/1 feature the model can use.

---

## Block 4 — BMI (Body Mass Index)

```python
for df in [train, test]:
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
```

### What it does
Creates a single number that combines Height and Weight into a measure of body density.

`BMI = Weight / Height²`

### Why
A player who weighs 130 kg but is very short is built differently from a player who weighs 130 kg and is very tall. BMI captures this ratio. It is already in the baseline model and is a standard physical feature for combine data, so we keep it.

---

## Block 5 — Label-Encoding Categorical Columns

```python
cat_cols = ['Player_Type', 'Position_Type', 'Position']
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[c], test[c]]).astype(str)
    le.fit(combined)
    train[c] = le.transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))
```

### What it does
Converts text columns like `"WR"`, `"QB"`, `"offense"` into numbers like `0`, `1`, `2`.

We fit the encoder on **both** train and test combined so that every possible text value gets a number, even if a value only appears in test but not in train.

### Why
Machine learning models work with numbers, not text. `LabelEncoder` builds a dictionary mapping each unique text value to a unique integer:

```
WR -> 0
QB -> 1
RB -> 2
...
```

### Why combine train + test before fitting
If we only used train to fit the encoder and test had a position not seen in train (e.g. a rare position), the encoder would crash or produce wrong results. Combining both ensures we handle every value correctly.

---

## Block 6 — Defining Feature Columns

```python
drop_cols    = ['Id', 'Drafted', 'School']
feature_cols = [c for c in train.columns if c not in drop_cols]

X      = train[feature_cols].copy()
y      = train['Drafted'].values
X_test = test[feature_cols].copy()
```

### What it does
- `drop_cols` — columns we do NOT feed as features to the model:
  - `Id` — just a row number, meaningless
  - `Drafted` — this is the *answer* we are trying to predict; including it would be cheating
  - `School` — excluded here because we handle it specially inside each fold (see Block 8)
- `feature_cols` — every other column becomes a feature (input to the model)
- `X` — the features for training rows
- `y` — the labels (0 or 1) for training rows
- `X_test` — the features for test rows (no labels)

---

## Block 7 — Smooth Target Encoder Function

```python
def smooth_encode(schools, fold_map, global_mean, count_map, smooth=10):
    enc = []
    for s in schools:
        if s in fold_map.index:
            n = count_map.get(s, 0)
            enc.append((n * fold_map[s] + smooth * global_mean) / (n + smooth))
        else:
            enc.append(global_mean)
    return np.array(enc)
```

### What it does
Converts each player's `School` name into a single number — the average draft rate for that school — but with a *smoothing* adjustment.

The formula:
```
encoded_value = (n * school_mean + 10 * global_mean) / (n + 10)
```

Where:
- `n` = how many players from that school are in the training fold
- `school_mean` = fraction of players from that school who were drafted
- `global_mean` = overall draft rate across all players
- `10` = the smoothing factor

### Why School matters
From the data audit, school draft rates range from 36% to 85%. A player from Alabama has a very different baseline chance of being drafted compared to a player from a small Division II school.

### Why smoothing is needed
Imagine a school with only 1 player in training, and that player was drafted. Without smoothing, the encoded value would be 1.0 (100% draft rate), which is misleading — we cannot trust a rate based on 1 sample. Smoothing pulls extreme estimates toward the overall average when there is little data, making the estimate more reliable.

### Why we cannot compute this before the fold loop (data leakage)
If we computed `School` averages on the full training set and then used those values during cross-validation, the validation fold's answer (`Drafted`) would have already influenced the encoding. The model would appear to perform better than it really is — this is called **data leakage**. We avoid it by computing the encoding *only on the training part of each fold*, never touching the validation rows.

---

## Block 8 — 5-Fold Cross-Validation Loop

```python
SEED     = 42
N_FOLDS  = 5
skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))
auc_scores = []
```

### What is cross-validation?
Instead of training on all data at once, we split the 2 781 training rows into 5 roughly equal chunks called *folds*.

```
Fold 1:  [TRAIN | TRAIN | TRAIN | TRAIN | VALIDATE]
Fold 2:  [TRAIN | TRAIN | TRAIN | VALIDATE | TRAIN]
Fold 3:  [TRAIN | TRAIN | VALIDATE | TRAIN | TRAIN]
Fold 4:  [TRAIN | VALIDATE | TRAIN | TRAIN | TRAIN]
Fold 5:  [VALIDATE | TRAIN | TRAIN | TRAIN | TRAIN]
```

Each fold, a different 20% of the data is held out for validation. We train on the other 80%, predict on the held-out 20%, and record how accurate we were.

### Why `StratifiedKFold`?
Normal splitting might accidentally put most drafted players in one fold. `StratifiedKFold` ensures each fold has the same ratio of drafted/undrafted as the overall dataset. This gives more stable and reliable performance estimates.

### Why `SEED = 42`?
The `random_state` makes the shuffle reproducible. Anyone who runs this script gets the exact same folds and the same results.

### `oof_preds` and `test_preds`
- `oof_preds` (Out-Of-Fold predictions) — after all 5 folds, every training row has exactly one prediction made when it was the held-out validation row. This gives us a fair, leak-free AUC score.
- `test_preds` — we average the test predictions from all 5 models (since each fold produces a slightly different model).

---

## Block 9 — Inside Each Fold

```python
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    fold_map    = train.iloc[tr_idx].groupby('School')['Drafted'].mean()
    global_mean = y_tr.mean()

    X_tr['School_enc'] = smooth_encode(train.iloc[tr_idx]['School'], fold_map, global_mean, train_school_counts)
    X_va['School_enc'] = smooth_encode(train.iloc[va_idx]['School'], fold_map, global_mean, train_school_counts)
```

### What it does
- `tr_idx` / `va_idx` — the row indices for this fold's training and validation sets
- We compute `fold_map` — the school draft rate using *only the training rows of this fold* (no leakage)
- `global_mean` — the overall draft rate in the training portion of this fold
- We add `School_enc` as a column to both `X_tr` and `X_va` using the safe encoding function

### Why copy?
`.copy()` ensures we do not accidentally modify the original `X` DataFrame when we add `School_enc`. Without it, changes in one fold could bleed into the next.

---

## Block 10 — Training the LightGBM Model

```python
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
    verbose=-1,
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)
```

### What is LightGBM?
LightGBM is a *gradient boosting* model. It builds hundreds of small decision trees one after another, where each new tree tries to fix the mistakes of all previous trees. The final prediction is the combined vote of all trees.

### Why LightGBM over the baseline Random Forest?
- Handles missing values **natively** (no need for imputation)
- Builds deeper, more expressive trees
- Generally achieves higher accuracy on tabular data

### Key hyperparameters explained
| Parameter | Value | Plain meaning |
|---|---|---|
| `n_estimators` | 1000 | Maximum number of trees to build |
| `learning_rate` | 0.05 | How much each new tree corrects the previous error — smaller = more careful |
| `num_leaves` | 63 | Maximum number of decision branches per tree — more = more complex patterns |
| `min_child_samples` | 20 | A leaf must have at least 20 players — prevents the model memorizing tiny groups |
| `feature_fraction` | 0.8 | Each tree only sees 80% of the features — reduces overfitting |
| `bagging_fraction` | 0.8 | Each tree only trains on 80% of the rows — reduces overfitting |
| `reg_alpha/lambda` | 0.1/1.0 | Regularization penalties — discourage overly complex trees |

### Early stopping
`early_stopping(50)` means: if the validation AUC has not improved for 50 consecutive trees, stop training. This prevents overfitting and saves time — we do not always need all 1000 trees.

---

## Block 11 — Recording Predictions

```python
oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
auc = roc_auc_score(y_va, oof_preds[va_idx])
auc_scores.append(auc)
print(f"Fold {fold+1}  AUC: {auc:.4f}")

X_te_fold = X_test.copy()
X_te_fold['School_enc'] = smooth_encode(test['School'], fold_map, global_mean, train_school_counts)
test_preds += model.predict_proba(X_te_fold)[:, 1] / N_FOLDS
```

### What it does
- `predict_proba(X_va)[:, 1]` — returns the model's probability that each player *was* drafted (column index 1 = positive class). Values are between 0 and 1.
- We store these probabilities in `oof_preds` at the validation row positions.
- `roc_auc_score` measures how well we ranked drafted vs undrafted players in this fold.
- For test, we add `1/5` of this fold's predictions to `test_preds`. After 5 folds, `test_preds` is the average over all 5 models — an *ensemble* that is more stable than any single fold.

### What is AUC?
AUC (Area Under the ROC Curve) measures ranking quality:
- **0.5** = random guessing (coin flip)
- **1.0** = perfect — every drafted player ranked above every undrafted player
- **0.82** (our result) = very good for this dataset

---

## Block 12 — Final OOF Score

```python
oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.4f}  |  Mean fold AUC: {np.mean(auc_scores):.4f} +/- {np.std(auc_scores):.4f}")
```

### What it does
Computes the AUC score across *all* out-of-fold predictions at once. This is the most reliable measure of model quality we have before submitting.

### Why trust OOF AUC more than the Public Leaderboard
The Public Leaderboard on the competition site is scored on only a fraction of the 696 test rows. A lucky or unlucky prediction on a few rows can move that number a lot. Our OOF score uses 2 781 rows with known answers, making it a far more stable and trustworthy estimate.

---

## Block 13 — Saving the Submission

```python
sub = pd.read_csv(PATH / "sample_submission.csv")
sub["Drafted"] = test_preds
sub.to_csv("submission_lgbm_phase0.csv", index=False)
print("Saved: submission_lgbm_phase0.csv")
```

### What it does
- Reads the `sample_submission.csv` provided by the competition — it has the right `Id` column in the right order.
- Replaces the placeholder `Drafted` column with our model's predictions.
- Saves the result as `submission_lgbm_phase0.csv`.

### Why use sample_submission as a template?
The competition requires a specific file format. Using the provided template guarantees our `Id` column matches exactly what the grader expects.

---

## Summary — Why This Works Better Than the Baseline

| Baseline (Random Forest) | Phase 0 (LightGBM) |
|---|---|
| Mean-imputes missing drills, losing signal | Keeps NaN natively + adds `_missing` flags |
| Drops `School` entirely | Encodes `School` as smooth target mean (fold-safe) |
| Shallow trees (max depth 5) | Deep trees with regularization |
| No early stopping | Stops when validation stops improving |
| Estimated OOF ~0.75–0.78 | Achieved OOF **0.8213** |

The three ideas that matter most, in order:
1. **Missing `Age` flag** — alone worth ~0.716 AUC
2. **All other missingness flags** — zero missing drills = very likely drafted
3. **School target encoding** — captures institutional draft pipeline strength
