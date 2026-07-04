# Phase B — Refactor to a package + persist a deployable model

## 1. Big Picture
Before this phase, the model logic lived *inside* `experiments/phase3_final.py`. That script does
something clever but impossible to deploy: it trains models **inside cross-validation folds** and
averages their predictions. There is never a single fitted model in memory that you could save and
reuse — the only output was `submission_FINAL.csv`.

Phase B fixes that in two moves:

1. **Refactor** the feature-engineering and data-loading logic out of the frozen phase script and into
   the reusable `src/nfl_draft/` package, so there is **one** definition of the pipeline, imported by
   both training and serving. This kills the #1 production-ML bug: training and serving computing
   features differently ("training/serving skew").
2. **Persist a deployable model** — a *new* artifact fit once on the **full** training set (not
   fold-internal), saved to disk with all its preprocessing state, so a single row can be scored later.

## 2. Core concepts, explained simply
- **Training vs. serving.** *Training* = fit models on many labelled rows at once. *Serving* = score
  one new, unlabelled player on demand. They must compute features identically or the model sees
  garbage at serve time.
- **Single source of truth.** The feature transforms live in exactly one file (`features/build.py`).
  Both `train.py` and `predict.py` import the same functions, so they can't drift apart.
- **Stateless transform vs. fitted state.** A *transform* (e.g. "compute BMI") is pure logic that runs
  the same on any row. *Fitted state* is something *learned from training data* — the mean drill value
  per position, how common each school is, the target-encoding map. Fitted state is learned **once** in
  `train.py`, **saved**, and **reloaded** by `predict.py`. Keeping the two separate is what lets a
  single serve-time row get encoded exactly as training saw it.
- **Model persistence (`joblib` / `.cbm`).** Saving a fitted model (and its preprocessing state) to
  disk so it can be loaded later without retraining. `joblib` is the standard for scikit-learn-style
  Python objects; CatBoost has its own native `.cbm` format via `save_model` / `load_model`.
- **Target encoding + smoothing.** Replace a category (School) with the average draft rate of players
  from that school. Rare schools (tiny sample) get pulled toward the global draft rate by a *smoothing*
  term (`smooth=10`) so a school with one lucky player doesn't get an extreme value.
- **The missingness rule (the model's edge).** A blank combine field must stay `NaN`, **never 0**. The
  `_missing` flags and `num_drills_missing` count only carry signal if blanks remain blank; filling a
  zero would both destroy the signal *and* lie (a 0-second 40-yard dash).
- **Unseen categories at serve time.** A user can enter a School or Position never seen in training.
  `LabelEncoder.transform` crashes on those, so `predict.py` maps unseen values to a fresh integer code;
  CatBoost handles unseen strings natively.

## 3. File-by-file
- `src/nfl_draft/data/load.py` — **new.** Three one-line readers (`load_train`, `load_test`,
  `load_sample_submission`) that pull paths from `config.py`. Isolates "where data comes from."
- `src/nfl_draft/features/build.py` — **new. The heart of the phase.** The transforms lifted from
  `phase3_final.py:39-117`: `add_missingness`, `add_composites`, `add_position_z`, `add_frequency`,
  `engineer_base` (runs the four in order), and `smooth_encode`. Each is a **stateless** function that
  takes any fitted maps it needs as arguments — so training and serving call the identical code.
- `src/nfl_draft/models/train.py` — **new.** Replaces the CV/seed loop. `_fit_preprocessing` learns all
  fitted state (position stats, frequency maps, label encoders, the full-train School target map).
  `_build_lgbm_matrix` / `_build_catboost_matrix` assemble each model's feature frame. `train_and_persist`
  fits one `LGBMClassifier` and one `CatBoostClassifier` on **full** train, prints an in-sample AUC
  smoke test, and saves three artifacts to `models/`:
  `lgbm_full.joblib`, `catboost_full.cbm`, `preprocess.joblib` (the state bundle + feature-column orders).
  Because there's no validation fold, there is **no early stopping** — we train the fixed `n_estimators`
  from the tuned params.
- `src/nfl_draft/models/predict.py` — **new.** `predict_one(player: dict) -> {probability, top_factors}`.
  Builds a 1-row frame (coercing blanks to `NaN`), runs `engineer_base`, applies the **saved** encoders
  and School map, reindexes to the saved column order, and blends `0.85*lgbm + 0.15*catboost`. Safe
  unseen-category handling via `_safe_transform`. `top_factors` returns `[]` for now — Phase C fills it
  with SHAP. Artifacts are lazily loaded once and cached.
- `tests/test_features.py`, `tests/test_predict.py` — **new.** Lock in the missingness invariants
  (flags fire, blanks stay `NaN`, composites don't zero-fill) and the serving contract (prob in `[0,1]`,
  blank-combine player scores lower than full-combine).
- `pyproject.toml`, `requirements.txt` — **edited.** Added `pytest==9.1.1` as a dev-only dependency and a
  `[tool.pytest.ini_options]` `testpaths = ["tests"]` entry.

## 4. Issues hit while building
- **In-sample AUC = 1.0000.** *What:* the training smoke test reported a perfect score. *Why:* a
  1000-tree boosted model scoring the very rows it trained on will memorize them — and the full-train
  School target-encoding feature effectively carries the label for training rows. *Fix:* nothing — this
  is expected. *Lesson:* **in-sample AUC is not a generalization estimate.** The honest number is the
  0.829 **out-of-fold** AUC from the experiments; you can never reproduce that from a single full-fit
  model. In-sample AUC is only a "does the pipeline run" smoke test here.
- **Forgot `import pandas as pd` in `train.py`.** *What:* `_fit_preprocessing` used `pd.concat` but the
  first draft imported only `numpy`. *Why:* a leftover from copying imports. *Fix:* import pandas, drop
  the unused numpy. *Lesson:* run the module immediately after writing to catch missing imports fast.
- **Unseen categories would crash serving.** *What:* `LabelEncoder.transform` raises `ValueError` on any
  category not seen during `fit`. *Why:* it has no code for a value it never learned. *Fix:* `predict.py`
  maps through `le.classes_` and sends unseen values to a fresh integer (`len(classes_)`). *Lesson:* the
  serving path faces inputs training never did — handle unseen categories and blank fields explicitly.
- **`no eval_set` for the full-train fit.** *What:* the fold code used early stopping against a
  validation fold; a full-train fit has no holdout. *Why:* early stopping needs a set to watch. *Fix:*
  drop early stopping and train the fixed `n_estimators`; drop CatBoost's `early_stopping_rounds` too.
  *Lesson:* a final "fit on everything" model trades early stopping for using every row.

## 5. Where things stand + what's next
The reusable package is real: `import nfl_draft` works, `python -m nfl_draft.models.train` writes three
artifacts, and `predict_one` scores a single player through the **same** pipeline training used —
verified by a train/serve **parity AUC of 1.0000** (serving reproduces training exactly) and a
full-combine player (0.759) scoring far above a blank-combine one (0.068). All 5 tests pass.

**Next — Phase C:** Weights & Biases experiment tracking (log the phase progression + Optuna trials +
upload the artifacts) and SHAP explainability (global importance bar → `reports/figures/shap_importance.png`,
and per-prediction top-3 factors wired into `predict_one`'s currently-empty `top_factors`).
