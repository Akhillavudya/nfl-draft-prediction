"""SHAP explainability: a global importance figure + per-prediction top factors."""
import joblib
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # draw to a file, not a screen (no GUI window needed)
import matplotlib.pyplot as plt

from nfl_draft.config import MODELS_DIR, REPORTS_FIGURES
from nfl_draft.data.load import load_train
from nfl_draft.models.train import _build_lgbm_matrix

_EXPLAINER = None


def get_explainer():
    """Build (once) a fast SHAP TreeExplainer around the persisted LightGBM model."""
    global _EXPLAINER
    if _EXPLAINER is None:
        lgbm = joblib.load(MODELS_DIR / "lgbm_full.joblib")
        _EXPLAINER = shap.TreeExplainer(lgbm)
    return _EXPLAINER


def _positive_class(shap_values):
    """Normalise SHAP output to a 2-D 'drafted-class' array across shap versions."""
    if isinstance(shap_values, list):        # older API: [class0, class1]
        return shap_values[1]
    if shap_values.ndim == 3:                # newer API: (rows, features, classes)
        return shap_values[..., 1]
    return shap_values                       # already (rows, features)


def _train_lgbm_matrix():
    """Rebuild the exact LightGBM training matrix via the shared train.py pipeline."""
    pp = joblib.load(MODELS_DIR / "preprocess.joblib")
    X, _ = _build_lgbm_matrix(load_train(), pp)
    return X


def save_global_importance(max_display=15):
    """Save a mean-|SHAP| bar chart of the most important features for the portfolio."""
    X = _train_lgbm_matrix()
    sv = _positive_class(get_explainer().shap_values(X))
    shap.summary_plot(sv, X, plot_type="bar", max_display=max_display, show=False)
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
    out = REPORTS_FIGURES / "shap_importance.png"
    plt.title("SHAP feature importance — NFL Draft model")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return out


def top_factors(lgbm_row, k=3):
    """Return the k features that moved one prediction most, each with sign and value."""
    sv = _positive_class(get_explainer().shap_values(lgbm_row))[0]
    feats = list(lgbm_row.columns)
    order = np.argsort(np.abs(sv))[::-1][:k]
    return [
        {"feature": feats[i],
         "value": None if np.isnan(v := lgbm_row.iloc[0, i]) else float(v),
         "shap": float(sv[i]),
         "direction": "increases" if sv[i] > 0 else "decreases"}
        for i in order
    ]


if __name__ == "__main__":
    save_global_importance()