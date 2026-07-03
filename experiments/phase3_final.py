import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

PATH    = Path("input")
N_FOLDS = 5
SEEDS   = [42, 123, 2025]

# Best Optuna params from Phase 2
LGBM_PARAMS = dict(
    n_estimators     = 1000,
    learning_rate    = 0.0645163786568917,
    num_leaves       = 41,
    max_depth        = 8,
    min_child_samples= 100,
    feature_fraction = 0.5571356598017494,
    bagging_fraction = 0.9242554434882404,
    bagging_freq     = 1,
    reg_alpha        = 0.0059161270802277665,
    reg_lambda       = 3.6116983011094534,
    verbose          = -1,
)

# Best blend weight from Phase 2
W_LGBM = 0.85
W_CB   = 1 - W_LGBM

train = pd.read_csv(PATH / "train.csv")
test  = pd.read_csv(PATH / "test.csv")

# ── Feature engineering ───────────────────────────────────────────────────────
drill_cols = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
              'Broad_Jump', 'Agility_3cone', 'Shuttle']

for df in [train, test]:
    for c in drill_cols:
        df[f'{c}_missing'] = df[c].isnull().astype(int)
    df['num_drills_missing'] = df[drill_cols].isnull().sum(axis=1)
    df['full_combine']       = (df['num_drills_missing'] == 0).astype(int)
    df['BMI']                = df['Weight'] / (df['Height'] ** 2)
    df['power']              = df['Vertical_Jump'] * df['Weight']
    df['speed_score']        = df['Weight'] / (df['Sprint_40yd'] ** 4 + 1e-9)
    df['agility_diff']       = df['Agility_3cone'] - df['Shuttle']
    df['jump_ratio']         = df['Broad_Jump'] / (df['Vertical_Jump'] + 1e-9)
    df['bench_per_kg']       = df['Bench_Press_Reps'] / (df['Weight'] + 1e-9)

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

school_freq = pd.concat([train['School'], test['School']]).value_counts()
for df in [train, test]:
    df['School_freq'] = df['School'].map(school_freq).fillna(0)

pos_freq = train['Position'].value_counts()
for df in [train, test]:
    df['Position_freq'] = df['Position'].map(pos_freq).fillna(0)

# Raw copy for CatBoost before label encoding
train_raw = train.copy()
test_raw  = test.copy()

cat_cols = ['Player_Type', 'Position_Type', 'Position']
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[c], test[c]]).astype(str)
    le.fit(combined)
    train[c] = le.transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))

drop_cols    = ['Id', 'Drafted', 'School']
feature_cols = [c for c in train.columns if c not in drop_cols]

X      = train[feature_cols].copy()
y      = train['Drafted'].values
X_test = test[feature_cols].copy()

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

# CatBoost setup
cb_drop   = ['Id', 'Drafted']
cb_feat   = [c for c in train_raw.columns if c not in cb_drop]
cat_feats = ['School', 'Player_Type', 'Position_Type', 'Position']

X_cb      = train_raw[cb_feat].copy()
X_cb_test = test_raw[cb_feat].copy()
for c in cat_feats:
    X_cb[c]      = X_cb[c].astype(str)
    X_cb_test[c] = X_cb_test[c].astype(str)

# ═════════════════════════════════════════════════════════════════════════════
# Seed averaging loop
# ═════════════════════════════════════════════════════════════════════════════
all_lgbm_oof  = []
all_lgbm_test = []
all_cb_oof    = []
all_cb_test   = []

for seed in SEEDS:
    print(f"\n{'='*55}")
    print(f"SEED {seed}")
    print(f"{'='*55}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    lgbm_oof  = np.zeros(len(train))
    lgbm_test = np.zeros(len(test))
    lgbm_aucs = []
    params    = {**LGBM_PARAMS, 'random_state': seed}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        fold_map    = train.iloc[tr_idx].groupby('School')['Drafted'].mean()
        global_mean = y_tr.mean()
        X_tr['School_enc'] = smooth_encode(train.iloc[tr_idx]['School'], fold_map, global_mean, train_school_counts)
        X_va['School_enc'] = smooth_encode(train.iloc[va_idx]['School'], fold_map, global_mean, train_school_counts)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        lgbm_oof[va_idx] = model.predict_proba(X_va)[:, 1]
        lgbm_aucs.append(roc_auc_score(y_va, lgbm_oof[va_idx]))

        X_te_fold = X_test.copy()
        X_te_fold['School_enc'] = smooth_encode(test['School'], fold_map, global_mean, train_school_counts)
        lgbm_test += model.predict_proba(X_te_fold)[:, 1] / N_FOLDS

    lgbm_oof_auc = roc_auc_score(y, lgbm_oof)
    print(f"  LGBM  OOF AUC: {lgbm_oof_auc:.4f}  (folds: {[round(a,4) for a in lgbm_aucs]})")
    all_lgbm_oof.append(lgbm_oof)
    all_lgbm_test.append(lgbm_test)

    # ── CatBoost ──────────────────────────────────────────────────────────────
    cb_oof  = np.zeros(len(train))
    cb_test = np.zeros(len(test))
    cb_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cb, y)):
        X_tr = X_cb.iloc[tr_idx]
        X_va = X_cb.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        model_cb = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=seed,
            verbose=0,
            eval_metric='AUC',
            early_stopping_rounds=50,
            cat_features=cat_feats,
        )
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        cb_oof[va_idx] = model_cb.predict_proba(X_va)[:, 1]
        cb_aucs.append(roc_auc_score(y_va, cb_oof[va_idx]))
        cb_test += model_cb.predict_proba(X_cb_test)[:, 1] / N_FOLDS

    cb_oof_auc = roc_auc_score(y, cb_oof)
    print(f"  CB    OOF AUC: {cb_oof_auc:.4f}  (folds: {[round(a,4) for a in cb_aucs]})")
    all_cb_oof.append(cb_oof)
    all_cb_test.append(cb_test)

# ═════════════════════════════════════════════════════════════════════════════
# Aggregate across seeds
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("FINAL AGGREGATION")
print(f"{'='*55}")

final_lgbm_oof  = np.mean(all_lgbm_oof,  axis=0)
final_lgbm_test = np.mean(all_lgbm_test, axis=0)
final_cb_oof    = np.mean(all_cb_oof,    axis=0)
final_cb_test   = np.mean(all_cb_test,   axis=0)

lgbm_avg_auc = roc_auc_score(y, final_lgbm_oof)
cb_avg_auc   = roc_auc_score(y, final_cb_oof)
print(f"Seed-avg LGBM  OOF AUC: {lgbm_avg_auc:.4f}")
print(f"Seed-avg CB    OOF AUC: {cb_avg_auc:.4f}")

# Blend
final_oof  = W_LGBM * final_lgbm_oof  + W_CB * final_cb_oof
final_test = W_LGBM * final_lgbm_test + W_CB * final_cb_test
blend_auc  = roc_auc_score(y, final_oof)
print(f"Seed-avg blend OOF AUC: {blend_auc:.4f}  (w_lgbm={W_LGBM}, w_cb={W_CB})")

print(f"\n--- Full Progression ---")
print(f"Phase 0  LGBM baseline       : 0.8213")
print(f"Phase 2  Tuned LGBM          : 0.8276")
print(f"Phase 2  Blend (85/15)       : 0.8292")
print(f"Phase 3  Seed-avg blend      : {blend_auc:.4f}  <-- FINAL")

# ── Sanity checks + save ───────────────────────────────────────────────────────
sub = pd.read_csv(PATH / "sample_submission.csv")
sub["Drafted"] = final_test

assert len(sub) == 696,                         "Row count mismatch"
assert sub["Drafted"].between(0, 1).all(),      "Predictions out of [0,1]"
assert list(sub.columns) == ["Id", "Drafted"],  "Column mismatch"

sub.to_csv("submission_FINAL.csv", index=False)
print("\nAll checks passed.")
print("Saved: submission_FINAL.csv")
