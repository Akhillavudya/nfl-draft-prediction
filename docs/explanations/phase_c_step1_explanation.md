# Phase C · Step 1 — Explainability with SHAP

## 1. Big Picture
The model already predicts well, but a probability alone is a black box. A reviewer (or a coach) will
ask *"why did it say that?"* Step 1 adds **SHAP**, which attributes every prediction to the features
that caused it. This does two jobs: it produces the **portfolio screenshot** (a global
feature-importance bar chart) and it lets the app show **per-prediction reasons**. Crucially, it turns
the project's headline claim — *"missing combine data is signal"* — from an assertion into something we
can literally point at on a chart.

## 2. Core concepts, explained simply
- **SHAP value** — for one prediction, a signed number per feature: how much that feature pushed the
  draft probability **up** (positive) or **down** (negative) versus an average player. It comes from
  game theory (Shapley values): treat each feature as a teammate and give it its fair share of the
  credit for the final score. The values *add up* to the gap between this prediction and the baseline.
- **`TreeExplainer`** — SHAP is slow in general, but tree models (LightGBM/CatBoost) have a fast
  **exact** algorithm. `shap.TreeExplainer(model)` uses it, so explaining costs almost nothing.
- **Global vs. local explanation** — *Global* = average `|SHAP|` across all training rows → "which
  features matter most overall" (the bar chart). *Local* = the top features for **one** player → what
  the UI shows ("blank Age drove this down").
- **Headless plotting (`matplotlib.use("Agg")`)** — "Agg" is a backend that renders straight to a PNG
  file with no on-screen window. It must be selected **before** `import matplotlib.pyplot`. We need it
  because this runs in scripts (and later on Modal's servers, which have no display at all).
- **JSON-legal values** — `NaN` is a valid Python float but **not** valid JSON. Any value that will
  later cross an HTTP boundary (Phase D) must be converted to `null` first.

## 3. File-by-file
- `src/nfl_draft/models/explain.py` — **new.** The single home for SHAP:
  - `get_explainer()` — lazily builds one `TreeExplainer` around the persisted `lgbm_full.joblib`.
  - `_positive_class()` — normalises SHAP output to the "drafted" class (the API returns different
    shapes across versions — see Issues).
  - `_train_lgbm_matrix()` — rebuilds the exact training matrix by **reusing** `train.py`'s
    `_build_lgbm_matrix`, so SHAP explains byte-for-byte what the model trained on (no drift).
  - `save_global_importance()` — mean-|SHAP| bar chart → `reports/figures/shap_importance.png`.
  - `top_factors(lgbm_row, k=3)` — the k most influential features for a single prediction, each with
    sign, value, and SHAP magnitude; missing values serialised as `None`.
- `src/nfl_draft/models/predict.py` — **edited.** `predict_one` now returns real `top_factors` by
  calling `explain.top_factors(lg)` on the same encoded row it feeds LightGBM. The placeholder
  `"top_factors": []` is gone.
- `pyproject.toml` / `requirements.txt` — **edited.** Pinned `shap==0.51.0` and `matplotlib==3.10.9`.

**What the figure showed:** `Age` (raw value) dominates, then `School_enc`, then `speed_score`, with
`Age_missing` landing top-6. So the missingness edge is clearly real — it just shares the spotlight
with the value itself, which is a *stronger* story than "only missingness matters." A full-combine test
player scored **0.966**; the same player with all drills blank scored **0.071**, and SHAP named
`Age=NaN` and `Age_missing=1` as the reasons.

## 4. Issues hit while building
- **`NaN` is not valid JSON.**
  - *What happened:* `predict_one` on an all-blank player returned `"value": NaN` in `top_factors`;
    `json.dumps` prints it, but it's illegal JSON.
  - *Why:* a blank combine field is deliberately kept as `NaN` (the missingness signal), and a strict
    JSON reader (FastAPI/browser/Gradio in Phase D) rejects `NaN` — and this is our *headline* input.
  - *Fix:* serialise a missing value as `None` → JSON `null`; verified with
    `json.dumps(out, allow_nan=False)` no longer raising.
  - *Lesson:* convert `NaN`/`Infinity` to `null` at the serialisation edge before anything hits HTTP.
- **SHAP output shape is version-dependent.**
  - *What happened:* a `UserWarning` — "LightGBM binary classifier ... output has changed to a list of
    ndarray."
  - *Fix/lesson:* the `_positive_class()` helper handles the list / 3-D / 2-D cases in one place, so
    the rest of the code is version-robust. Normalise a library's quirky return type once.
- **Matplotlib backend.** *Lesson:* set `matplotlib.use("Agg")` **before** importing `pyplot` in any
  save-only (headless) script, or it may try to open a GUI window and crash.
- **`.gitignore` has no inline comments.**
  - *What happened:* `reports/figures/*.png` ignores generated figures, but `shap_importance.png` is a
    deliverable we *want* tracked. Adding `!reports/figures/shap_importance.png   # portfolio` on one
    line failed — the file was still ignored.
  - *Why:* in `.gitignore`, a `#` is only a comment at the **start** of a line; mid-line it becomes part
    of the pattern, so the negation matched `...png   # portfolio` (a file that doesn't exist).
  - *Fix/lesson:* put the comment on its own line above the rule. Verify negations with
    `git check-ignore <path>`.

## 5. Where things stand + what's next
The model is now **explainable**: a global SHAP figure for the portfolio, and per-prediction
`top_factors` wired into `predict.py` for the UI. All tests still pass. **Next (Step 2):** log this
figure — plus the phase AUC progression, the 50 Optuna trials, and the model artifacts — to a public
Weights & Biases run for the CV link.
