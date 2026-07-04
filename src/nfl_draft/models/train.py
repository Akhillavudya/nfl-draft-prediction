"""Fit one LightGBM + one CatBoost on the full training set and persist all serving state."""
import warnings

import joblib
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from nfl_draft.config import (
    MODELS_DIR, SEED, TARGET, DRILL_COLS, DRILL_VALS, CAT_COLS, CAT_FEATS,
    DROP_COLS, LGBM_PARAMS, CATBOOST_PARAMS, W_LGBM, W_CB, SMOOTH,
)
from nfl_draft.data.load import load_train, load_test
from nfl_draft.features.build import engineer_base, smooth_encode

warnings.filterwarnings("ignore")


def _fit_preprocessing(train, test):
    """Learn every fitted map (position stats, frequencies, encoders, School target map)."""
    pos_stats = train.groupby("Position")[DRILL_VALS].agg(["mean", "std"])
    school_freq = pd.concat([train["School"], test["School"]]).value_counts()
    pos_freq = train["Position"].value_counts()

    label_encoders = {}
    for c in CAT_COLS:
        le = LabelEncoder()
        le.fit(pd.concat([train[c], test[c]]).astype(str))
        label_encoders[c] = le

    school_map = train.groupby("School")[TARGET].mean()
    global_mean = float(train[TARGET].mean())
    train_school_counts = train["School"].value_counts().to_dict()

    return dict(
        pos_stats=pos_stats, school_freq=school_freq, pos_freq=pos_freq,
        label_encoders=label_encoders, school_map=school_map,
        global_mean=global_mean, train_school_counts=train_school_counts,
        smooth=SMOOTH, cat_feats=CAT_FEATS, drill_cols=DRILL_COLS, drill_vals=DRILL_VALS,
        w_lgbm=W_LGBM, w_cb=W_CB,
    )


def _build_lgbm_matrix(df, pp):
    """Engineer features, add the saved School_enc, label-encode cats → LightGBM feature frame."""
    eng = engineer_base(df, pp["pos_stats"], pp["school_freq"], pp["pos_freq"])
    eng["School_enc"] = smooth_encode(
        df["School"], pp["school_map"], pp["global_mean"], pp["train_school_counts"], pp["smooth"]
    )
    for c in CAT_COLS:
        eng[c] = pp["label_encoders"][c].transform(eng[c].astype(str))
    feature_cols = [c for c in eng.columns if c not in DROP_COLS]
    return eng[feature_cols], feature_cols


def _build_catboost_matrix(df, pp):
    """Engineer features and keep raw string categoricals → CatBoost feature frame."""
    eng = engineer_base(df, pp["pos_stats"], pp["school_freq"], pp["pos_freq"])
    for c in CAT_FEATS:
        eng[c] = eng[c].astype(str)
    cb_feat = [c for c in eng.columns if c not in ["Id", TARGET]]
    return eng[cb_feat], cb_feat


def train_and_persist():
    """Fit both models on full train, print an in-sample AUC, and save all three artifacts."""
    train = load_train()
    test = load_test()
    y = train[TARGET].values

    pp = _fit_preprocessing(train, test)

    X_lgbm, feature_cols = _build_lgbm_matrix(train, pp)
    lgbm = lgb.LGBMClassifier(**{**LGBM_PARAMS, "random_state": SEED})
    lgbm.fit(X_lgbm, y)

    X_cb, cb_feat = _build_catboost_matrix(train, pp)
    cb_params = {k: v for k, v in CATBOOST_PARAMS.items() if k != "early_stopping_rounds"}
    catboost = CatBoostClassifier(random_seed=SEED, cat_features=CAT_FEATS, **cb_params)
    catboost.fit(X_cb, y)

    lgbm_p = lgbm.predict_proba(X_lgbm)[:, 1]
    cb_p = catboost.predict_proba(X_cb)[:, 1]
    blend = W_LGBM * lgbm_p + W_CB * cb_p
    print(f"In-sample blend AUC: {roc_auc_score(y, blend):.4f}")

    pp["feature_cols"] = feature_cols
    pp["cb_feat"] = cb_feat

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm, MODELS_DIR / "lgbm_full.joblib")
    catboost.save_model(str(MODELS_DIR / "catboost_full.cbm"))
    joblib.dump(pp, MODELS_DIR / "preprocess.joblib")
    print(f"Saved 3 artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    train_and_persist()
