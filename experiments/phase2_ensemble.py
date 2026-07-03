import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import warnings
from pathlib import Path
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

PATH  = Path("input")
SEED  = 42
N_FOLDS = 5

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

# ── Physical composites ────────────────────────────────────────────────────────
for df in [train, test]:
    df['power']        = df['Vertical_Jump'] * df['Weight']
    df['speed_score']  = df['Weight'] / (df['Sprint_40yd'] ** 4 + 1e-9)
    df['agility_diff'] = df['Agility_3cone'] - df['Shuttle']
    df['jump_ratio']   = df['Broad_Jump'] / (df['Vertical_Jump'] + 1e-9)
    df['bench_per_kg'] = df['Bench_Press_Reps'] / (df['Weight'] + 1e-9)

# ── Position-relative z-scores ────────────────────────────────────────────────
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

# ── Frequency encodings ───────────────────────────────────────────────────────
school_freq = pd.concat([train['School'], test['School']]).value_counts()
for df in [train, test]:
    df['School_freq'] = df['School'].map(school_freq).fillna(0)

pos_freq = train['Position'].value_counts()
for df in [train, test]:
    df['Position_freq'] = df['Position'].map(pos_freq).fillna(0)

# ── Label-encode categoricals (for LightGBM) ──────────────────────────────────
# Keep a raw copy for CatBoost before encoding
train_raw = train.copy()
test_raw  = test.copy()

cat_cols = ['Player_Type', 'Position_Type', 'Position']
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[c], test[c]]).astype(str)
    le.fit(combined)
    train[c] = le.transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))

# ── Shared helpers ────────────────────────────────────────────────────────────
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

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ═════════════════════════════════════════════════════════════════════════════
# PART A — Optuna hyperparameter search for LightGBM
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("PART A: Optuna tuning (50 trials) ...")
print("=" * 55)

def run_lgbm_cv(params):
    oof = np.zeros(len(train))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        fold_map    = train.iloc[tr_idx].groupby('School')['Drafted'].mean()
        global_mean = y_tr.mean()
        X_tr['School_enc'] = smooth_encode(train.iloc[tr_idx]['School'], fold_map, global_mean, train_school_counts)
        X_va['School_enc'] = smooth_encode(train.iloc[va_idx]['School'], fold_map, global_mean, train_school_counts)

        model = lgb.LGBMClassifier(**params, verbose=-1)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y, oof)

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
        random_state     = SEED,
    )
    return run_lgbm_cv(params)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print(f"\nBest Optuna OOF AUC: {study.best_value:.4f}")
print(f"Best params: {best_params}")

# ═════════════════════════════════════════════════════════════════════════════
# PART B — Final LightGBM OOF + test preds with best params
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("PART B: Final LightGBM with tuned params ...")
print("=" * 55)

lgbm_params = dict(
    n_estimators     = 1000,
    learning_rate    = best_params["lr"],
    num_leaves       = best_params["num_leaves"],
    max_depth        = best_params["max_depth"],
    min_child_samples= best_params["min_child"],
    feature_fraction = best_params["feat_frac"],
    bagging_fraction = best_params["bag_frac"],
    bagging_freq     = 1,
    reg_alpha        = best_params["alpha"],
    reg_lambda       = best_params["lambda"],
    random_state     = SEED,
    verbose          = -1,
)

lgbm_oof   = np.zeros(len(train))
lgbm_test  = np.zeros(len(test))
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

    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    lgbm_oof[va_idx] = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, lgbm_oof[va_idx])
    auc_scores.append(auc)
    print(f"  Fold {fold+1}  AUC: {auc:.4f}")

    X_te_fold = X_test.copy()
    X_te_fold['School_enc'] = smooth_encode(test['School'], fold_map, global_mean, train_school_counts)
    lgbm_test += model.predict_proba(X_te_fold)[:, 1] / N_FOLDS

lgbm_oof_auc = roc_auc_score(y, lgbm_oof)
print(f"\nTuned LGBM  OOF AUC: {lgbm_oof_auc:.4f}  |  Mean: {np.mean(auc_scores):.4f} +/- {np.std(auc_scores):.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# PART C — CatBoost (School as native categorical)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("PART C: CatBoost with native School encoding ...")
print("=" * 55)

cb_drop    = ['Id', 'Drafted']
cb_feat    = [c for c in train_raw.columns if c not in cb_drop]
cat_feats  = ['School', 'Player_Type', 'Position_Type', 'Position']

X_cb      = train_raw[cb_feat].copy()
X_cb_test = test_raw[cb_feat].copy()

# CatBoost needs string categoricals
for c in cat_feats:
    X_cb[c]      = X_cb[c].astype(str)
    X_cb_test[c] = X_cb_test[c].astype(str)

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
        random_seed=SEED,
        verbose=0,
        eval_metric='AUC',
        early_stopping_rounds=50,
        cat_features=cat_feats,
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    cb_oof[va_idx] = model_cb.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, cb_oof[va_idx])
    cb_aucs.append(auc)
    print(f"  Fold {fold+1}  AUC: {auc:.4f}")

    cb_test += model_cb.predict_proba(X_cb_test)[:, 1] / N_FOLDS

cb_oof_auc = roc_auc_score(y, cb_oof)
print(f"\nCatBoost    OOF AUC: {cb_oof_auc:.4f}  |  Mean: {np.mean(cb_aucs):.4f} +/- {np.std(cb_aucs):.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# PART D — Rank-average ensemble + optimal blend weight
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("PART D: Rank-average ensemble ...")
print("=" * 55)

def rank_avg(preds_list):
    n = len(preds_list[0])
    ranked = [rankdata(p) / n for p in preds_list]
    return np.mean(ranked, axis=0)

# Grid search blend weight on OOF
best_w, best_blend_auc = None, 0
for w in np.arange(0.0, 1.05, 0.05):
    blended = w * lgbm_oof + (1 - w) * cb_oof
    a = roc_auc_score(y, blended)
    if a > best_blend_auc:
        best_blend_auc, best_w = a, w

print(f"Best blend weight (LGBM): {best_w:.2f}  OOF AUC: {best_blend_auc:.4f}")

# Final test predictions using best weight
final_test = best_w * lgbm_test + (1 - best_w) * cb_test

# Also compute rank-avg for reference
rank_oof  = rank_avg([lgbm_oof, cb_oof])
rank_auc  = roc_auc_score(y, rank_oof)
rank_test = rank_avg([lgbm_test, cb_test])
print(f"Rank-avg blend          OOF AUC: {rank_auc:.4f}")

# Pick the better ensemble method
if rank_auc >= best_blend_auc:
    final_test = rank_test
    print("Using rank-average for submission.")
else:
    print(f"Using weighted blend (w_lgbm={best_w:.2f}) for submission.")

print("\n--- Summary ---")
print(f"Phase 0 baseline        OOF AUC: 0.8213")
print(f"Tuned LGBM              OOF AUC: {lgbm_oof_auc:.4f}")
print(f"CatBoost                OOF AUC: {cb_oof_auc:.4f}")
print(f"Best blend              OOF AUC: {max(best_blend_auc, rank_auc):.4f}")

# ── Submission ─────────────────────────────────────────────────────────────────
sub = pd.read_csv(PATH / "sample_submission.csv")
sub["Drafted"] = final_test
sub.to_csv("submission_phase2_ensemble.csv", index=False)
print("\nSaved: submission_phase2_ensemble.csv")
