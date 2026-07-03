import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

PATH = Path("input")
train = pd.read_csv(PATH / "train.csv")
test  = pd.read_csv(PATH / "test.csv")

# ── Missingness flags ──────────────────────────────────────────────────────────
drill_cols = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
              'Broad_Jump', 'Agility_3cone', 'Shuttle']

for df in [train, test]:
    for c in drill_cols:
        df[f'{c}_missing'] = df[c].isnull().astype(int)
    df['num_drills_missing'] = df[drill_cols].isnull().sum(axis=1)
    df['full_combine']       = (df['num_drills_missing'] == 0).astype(int)

# ── BMI ────────────────────────────────────────────────────────────────────────
for df in [train, test]:
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# ── Phase 1A: Physical composites ─────────────────────────────────────────────
for df in [train, test]:
    df['power']        = df['Vertical_Jump'] * df['Weight']
    df['speed_score']  = df['Weight'] / (df['Sprint_40yd'] ** 4 + 1e-9)
    df['agility_diff'] = df['Agility_3cone'] - df['Shuttle']
    df['jump_ratio']   = df['Broad_Jump'] / (df['Vertical_Jump'] + 1e-9)
    df['bench_per_kg'] = df['Bench_Press_Reps'] / (df['Weight'] + 1e-9)

# ── Phase 1B: Position-relative drill z-scores ────────────────────────────────
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

# ── Phase 1C: Frequency encodings ─────────────────────────────────────────────
school_freq = pd.concat([train['School'], test['School']]).value_counts()
for df in [train, test]:
    df['School_freq'] = df['School'].map(school_freq).fillna(0)

pos_freq = train['Position'].value_counts()
for df in [train, test]:
    df['Position_freq'] = df['Position'].map(pos_freq).fillna(0)

# ── Label-encode categoricals ─────────────────────────────────────────────────
cat_cols = ['Player_Type', 'Position_Type', 'Position']
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[c], test[c]]).astype(str)
    le.fit(combined)
    train[c] = le.transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))

# ── Feature columns ────────────────────────────────────────────────────────────
drop_cols    = ['Id', 'Drafted', 'School']
feature_cols = [c for c in train.columns if c not in drop_cols]

X      = train[feature_cols].copy()
y      = train['Drafted'].values
X_test = test[feature_cols].copy()

# ── Smooth target encoder ─────────────────────────────────────────────────────
train_school_counts = train['School'].value_counts().to_dict()

def smooth_encode(schools, fold_map, global_mean, count_map, smooth=10):
    enc = []
    for s in schools:
        if s in fold_map.index:
            n = count_map.get(s, 0)
            enc.append((n * fold_map[s] + smooth * global_mean) / (n + smooth))
        else:
            enc.append(global_mean)
    return np.array(enc)

# ── 5-fold OOF loop ────────────────────────────────────────────────────────────
SEED    = 42
N_FOLDS = 5
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))
auc_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    fold_map    = train.iloc[tr_idx].groupby('School')['Drafted'].mean()
    global_mean = y_tr.mean()

    X_tr['School_enc'] = smooth_encode(train.iloc[tr_idx]['School'], fold_map, global_mean, train_school_counts)
    X_va['School_enc'] = smooth_encode(train.iloc[va_idx]['School'], fold_map, global_mean, train_school_counts)

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

    oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, oof_preds[va_idx])
    auc_scores.append(auc)
    print(f"Fold {fold+1}  AUC: {auc:.4f}")

    X_te_fold = X_test.copy()
    X_te_fold['School_enc'] = smooth_encode(test['School'], fold_map, global_mean, train_school_counts)
    test_preds += model.predict_proba(X_te_fold)[:, 1] / N_FOLDS

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.4f}  |  Mean fold AUC: {np.mean(auc_scores):.4f} +/- {np.std(auc_scores):.4f}")

# ── Submission ─────────────────────────────────────────────────────────────────
sub = pd.read_csv(PATH / "sample_submission.csv")
sub["Drafted"] = test_preds
sub.to_csv("submission_lgbm_phase1.csv", index=False)
print("Saved: submission_lgbm_phase1.csv")
