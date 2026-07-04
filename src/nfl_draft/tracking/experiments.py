"""Weights & Biases run: log the phase progression, re-run Optuna, upload artifacts + SHAP."""
import os
import warnings

import numpy as np
import optuna
import wandb
import lightgbm as lgb
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from nfl_draft.config import (
    SEED, N_FOLDS, TARGET, MODELS_DIR, CAT_COLS, DROP_COLS,
)
from nfl_draft.data.load import load_train, load_test
from nfl_draft.features.build import engineer_base, smooth_encode
from nfl_draft.models.train import _fit_preprocessing
from nfl_draft.models.explain import save_global_importance

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# The known OOF AUC at each modelling milestone (from experiments/phase3_final.py).
PHASE_PROGRESSION = [
    ("Phase 0", "LGBM baseline", 0.8213),
    ("Phase 2", "Tuned LGBM (Optuna)", 0.8276),
    ("Phase 2", "Blend 85/15", 0.8292),
    ("Phase 3", "Seed-averaged blend", 0.8290),
]


def _base_lgbm_matrix():
    """Build the label-encoded feature frame WITHOUT School_enc (added per fold, leak-safe)."""
    train, test = load_train(), load_test()
    pp = _fit_preprocessing(train, test)
    eng = engineer_base(train, pp["pos_stats"], pp["school_freq"], pp["pos_freq"])
    for c in CAT_COLS:
        eng[c] = pp["label_encoders"][c].transform(eng[c].astype(str))
    base_cols = [c for c in eng.columns if c not in DROP_COLS]
    return eng[base_cols].copy(), train, pp


def run_lgbm_cv(params, X, train, pp, y, skf):
    """5-fold OOF AUC with fold-internal School target encoding (the leak-safe recipe)."""
    oof = np.zeros(len(train))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        fold_map = train.iloc[tr_idx].groupby("School")[TARGET].mean()
        gmean = y[tr_idx].mean()
        X_tr["School_enc"] = smooth_encode(train.iloc[tr_idx]["School"], fold_map, gmean, pp["train_school_counts"])
        X_va["School_enc"] = smooth_encode(train.iloc[va_idx]["School"], fold_map, gmean, pp["train_school_counts"])
        model = lgb.LGBMClassifier(**params, verbose=-1)
        model.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y, oof)


def _objective(X, train, pp, y, skf):
    """Wrap run_lgbm_cv into an Optuna objective over the LightGBM search space."""
    def objective(trial):
        params = dict(
            n_estimators=1000,
            learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child", 5, 100),
            feature_fraction=trial.suggest_float("feat_frac", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bag_frac", 0.5, 1.0),
            bagging_freq=1,
            reg_alpha=trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            random_state=SEED,
        )
        return run_lgbm_cv(params, X, train, pp, y, skf)
    return objective


def main(n_trials=50):
    """Run the whole tracked experiment end to end and push it to a public W&B project."""
    load_dotenv()
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(project="nfl-draft-prediction", name="phaseC-tracking",
                     config={"n_trials": n_trials, "seed": SEED, "n_folds": N_FOLDS})

    table = wandb.Table(columns=["phase", "description", "oof_auc"], data=[list(r) for r in PHASE_PROGRESSION])
    wandb.log({"phase_progression": table})

    X, train, pp = _base_lgbm_matrix()
    y = train[TARGET].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))

    def log_trial(study, trial):
        wandb.log({"trial": trial.number, "trial_auc": trial.value, "best_auc_so_far": study.best_value})

    study.optimize(_objective(X, train, pp, y, skf), n_trials=n_trials, callbacks=[log_trial])

    wandb.run.summary["best_oof_auc"] = study.best_value
    wandb.run.summary.update({f"best_{k}": v for k, v in study.best_params.items()})
    print(f"Best OOF AUC: {study.best_value:.4f}")

    fig_path = save_global_importance()
    wandb.log({"shap_importance": wandb.Image(str(fig_path))})

    artifact = wandb.Artifact("nfl-draft-model", type="model")
    for f in ("lgbm_full.joblib", "catboost_full.cbm", "preprocess.joblib"):
        artifact.add_file(str(MODELS_DIR / f))
    wandb.log_artifact(artifact)

    run.finish()
    print(f"W&B run: {run.url}")


if __name__ == "__main__":
    main()
