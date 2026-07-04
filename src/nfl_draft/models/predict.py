"""Serving path: turn one raw player dict into a blended draft probability."""
import joblib
import pandas as pd
from catboost import CatBoostClassifier

from nfl_draft.config import MODELS_DIR, CAT_COLS, CAT_FEATS, DRILL_COLS, TARGET
from nfl_draft.features.build import engineer_base, smooth_encode

# Numeric raw fields that must be coerced (blank → NaN, never 0).
_NUMERIC_RAW = ["Height", "Weight"] + DRILL_COLS
# Every raw column the feature pipeline reads from a player.
_RAW_COLS = _NUMERIC_RAW + ["School", "Player_Type", "Position_Type", "Position"]

_ARTIFACTS = None


def _load():
    """Lazily load the three artifacts once and cache them for later calls."""
    global _ARTIFACTS
    if _ARTIFACTS is None:
        lgbm = joblib.load(MODELS_DIR / "lgbm_full.joblib")
        catboost = CatBoostClassifier()
        catboost.load_model(str(MODELS_DIR / "catboost_full.cbm"))
        pp = joblib.load(MODELS_DIR / "preprocess.joblib")
        _ARTIFACTS = (lgbm, catboost, pp)
    return _ARTIFACTS


def _safe_transform(le, values):
    """Label-encode, sending any category the encoder never saw to a fresh integer code."""
    mapping = {cls: i for i, cls in enumerate(le.classes_)}
    n = len(le.classes_)
    return values.astype(str).map(lambda v: mapping.get(v, n)).astype(int)


def _to_frame(player):
    """Build a 1-row frame with all raw columns present and numerics coerced (blank → NaN)."""
    df = pd.DataFrame([player]).reindex(columns=_RAW_COLS)
    for c in _NUMERIC_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def predict_one(player):
    """Blend LightGBM + CatBoost on one player dict → {probability, top_factors}."""
    lgbm, catboost, pp = _load()
    df = _to_frame(player)

    eng = engineer_base(df, pp["pos_stats"], pp["school_freq"], pp["pos_freq"])

    lg = eng.copy()
    lg["School_enc"] = smooth_encode(
        df["School"], pp["school_map"], pp["global_mean"], pp["train_school_counts"], pp["smooth"]
    )
    for c in CAT_COLS:
        lg[c] = _safe_transform(pp["label_encoders"][c], lg[c])
    lg = lg.reindex(columns=pp["feature_cols"])
    lgbm_p = lgbm.predict_proba(lg)[:, 1][0]

    cb = eng.copy()
    for c in CAT_FEATS:
        cb[c] = cb[c].astype(str)
    cb = cb.reindex(columns=pp["cb_feat"])
    cb_p = catboost.predict_proba(cb)[:, 1][0]

    prob = pp["w_lgbm"] * lgbm_p + pp["w_cb"] * cb_p
    return {"probability": float(prob), "top_factors": []}
