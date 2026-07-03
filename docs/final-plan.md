# NFL Draft Prediction — CV-Ready Implementation Plan

> Deployment & portfolio plan for turning this competition script into a shipped, tracked,
> explainable model. Planning artifact only — implement later.

---

## Context — why this work

The NFL Draft Prediction project is the **#2 project on the DS / AI-engineer CV (rated 9/10)**.
Its analytical content is already strong: a staged LightGBM + CatBoost ensemble at
**0.829 OOF ROC AUC**, whose edge is treating *missing* combine data as signal and leak-safe
per-fold target encoding of `School`.

Three things stop it from being CV/portfolio-ready:

1. **It ships nothing runnable** — every phase script trains inside CV folds and averages
   predictions. `submission_FINAL.csv` (696 rows of probabilities) is the only output.
   **No fitted model or preprocessing state is ever saved to disk.** So "deploy the model"
   first requires adding a *fit-on-full-train-and-persist* step — this is the real MLOps work.
2. **No deployment, no tracking, no explainability** — nothing that shows "can ship + can
   interpret a model," which is exactly the gap across the rest of the portfolio.
3. **Repo hygiene** — not a git repo, `venv/` sits in-folder, no `requirements.txt`, no real
   README, and the leaderboard rank is still pending.

**Goal / outcome:** turn a batch-CSV competition script into a *shipped, tracked, explainable*
model — a **Modal serverless endpoint + small Gradio UI + Weights & Biases tracking + SHAP** —
while deliberately **avoiding another Streamlit/HF app** (the portfolio's monoculture problem).

**Decisions locked:**
- Deployment scope = **Modal API + Gradio UI** (with SHAP + W&B).
- Leaderboard rank/certificate is **not out yet (~mid-July 2026)** → treat as a clearly-marked
  placeholder that will lead the CV bullet once it lands; until then lead with 0.829 OOF AUC.

---

## Skills this adds (the point of the exercise)

| Area | Today | After this plan |
|---|---|---|
| ML modeling | GBDT ensemble, Optuna, CV rigor | unchanged (already strong) |
| **Model persistence** | none (model exists only mid-run) | `joblib` artifacts: models + all preprocessing state |
| **Serverless deploy** | none | **Modal** endpoint (new modern platform) |
| **Inference eng.** | none | single-row feature pipeline + correct missing-data handling |
| **Experiment tracking** | in-memory Optuna only | **Weights & Biases** run: trials + phase AUCs + artifacts |
| **Explainability** | none | **SHAP** global bar + per-prediction top factors |
| **Front-end for ML** | none | small **Gradio** app (form → prob → SHAP) |
| **Reproducibility** | seeds set, deps unpinned | git + `requirements.txt` + README + `.gitignore` |

---

## Key technical finding that shapes the plan

The served model must be a **new artifact fit on the full training set** — distinct from the
CV/seed-averaged pipeline that produced the submission. The submission uses fold-internal
encodings and seed-averaged fold models (great for scoring, impossible to serve as one object).
For serving, fit once on all of `train.csv` and persist everything inference needs.

**Artifacts to persist (all reconstructable from `phase3_final.py` logic):**
- `lgbm_full.joblib` — one `LGBMClassifier` fit on full train (best params, `phase3_final.py:18-30`).
- `catboost_full.cbm` — one `CatBoostClassifier` fit on full train (native `School`).
- `preprocess.joblib` — a dict bundling:
  - `pos_stats` (Position drill mean/std, `phase3_final.py:57`) for z-scores
  - `school_freq`, `pos_freq` maps (`:69-75`)
  - `LabelEncoder`s for `Player_Type`, `Position_Type`, `Position` (LGBM path)
  - full-train **smoothed School target-encoding** map + `global_mean` + `train_school_counts` + `smooth=10`
  - `feature_cols` order, `drill_cols`, `cat_feats`, blend weight `W_LGBM=0.85`
- **Inference-time missing-data rule (critical correctness point):** a blank combine field must
  stay `NaN`, never `0`, so the `_missing` flags and `num_drills_missing` fire correctly — the
  whole model edge depends on this.

---

## Target repository structure (industry layout — the restructure)

Goal: anyone (recruiter, reviewer, teammate) can open the repo and immediately know where
data, reusable code, experiments, artifacts, and the app live. Convention =
**Cookiecutter Data Science + `src/` package layout**: raw data separated from code, one
importable package (`nfl_draft`) holding the reusable logic, the phase scripts preserved as
`experiments/` history, and all generated artifacts/reports gitignored.

```
nfl-draft-prediction/
├── README.md                     # real project README (Phase A)
├── pyproject.toml                # package metadata + pinned deps (installable: pip install -e .)
├── requirements.txt              # convenience pinned list mirroring pyproject
├── .gitignore
├── Makefile                      # make setup / train / track / app  (optional, nice-to-have)
│
├── data/
│   ├── raw/                      # train.csv, test.csv, sample_submission.csv   (gitignored)
│   └── processed/                # cached engineered frames, if any             (gitignored)
│
├── notebooks/
│   ├── 0_eda.ipynb               # was baseline.ipynb
│   ├── README.ipynb              # competition-provided notebook (kept as-is)
│   └── tutorials/                # was advanced_techniques/*.ipynb
│
├── experiments/                  # "how it was built" — kept verbatim, NOT refactored
│   ├── phase0_lgbm.py
│   ├── phase1_lgbm.py
│   ├── phase2_ensemble.py
│   └── phase3_final.py
│
├── src/
│   └── nfl_draft/                # the reusable package (importable everywhere)
│       ├── __init__.py
│       ├── config.py             # PATHS, SEEDS, N_FOLDS, LGBM_PARAMS, W_LGBM, column lists
│       ├── data/
│       │   ├── __init__.py
│       │   └── load.py           # load_train / load_test / load_sample_submission
│       ├── features/
│       │   ├── __init__.py
│       │   └── build.py          # missingness flags, composites, position z-scores,
│       │                         #   frequency encodings, smooth_encode()  (single source of truth)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train.py          # fit-on-full-train + persist artifacts   (was train_final.py)
│       │   └── predict.py        # predict_one(player: dict) -> {prob, top_factors}
│       └── tracking/
│           ├── __init__.py
│           └── experiments.py    # W&B run + SHAP figure                   (was track_experiments.py)
│
├── app/
│   ├── modal_app.py              # Modal serverless endpoint (Phase D)
│   └── gradio_app.py             # Gradio UI, thin client to Modal (Phase E; was app.py)
│
├── models/                       # persisted model artifacts               (generated, gitignored)
│   ├── lgbm_full.joblib
│   ├── catboost_full.cbm
│   └── preprocess.joblib
│
├── reports/
│   └── figures/
│       └── shap_importance.png   # portfolio screenshot (Phase C)
│
├── docs/
│   ├── final-plan.md             # this plan
│   ├── IMPLEMENTATION_PLAN.md    # moved from root
│   ├── extract.md                # moved from root
│   └── explanations/
│       └── phase0_explanations.md
│
├── references/
│   └── competition_tutorial.pdf  # moved from root
│
└── tests/
    ├── test_features.py          # missingness flags fire; blank stays NaN, never 0
    └── test_predict.py           # predict_one in [0,1]; blank-combine player < full-combine player
```

**Why this shape (what each choice signals):**
- **`src/nfl_draft/` package** — the feature-engineering + encoding logic (`phase3_final.py:39-117`)
  becomes `features/build.py`, imported by `models/train.py`, `models/predict.py`, and the app.
  One definition of the pipeline → no drift between training and serving (the classic serving bug).
- **`experiments/` keeps the phase scripts untouched** — they document the iteration story and the
  per-phase AUC gains; they are history, not the maintained code path.
- **`data/raw/` + gitignore** — never commit the CSVs/artifacts; the repo stays small and clean.
- **`config.py` as the single knob panel** — seeds, params, paths, column lists in one place
  instead of copied across four scripts.

**Move mapping (Phase A executes this — pure `git mv`, no logic change):**

| From (today) | To |
|---|---|
| `input/*.csv` | `data/raw/` |
| `phase0..3_*.py` | `experiments/` |
| `baseline.ipynb` | `notebooks/0_eda.ipynb` |
| `README.ipynb` | `notebooks/README.ipynb` |
| `advanced_techniques/*.ipynb` | `notebooks/tutorials/` |
| `IMPLEMENTATION_PLAN.md`, `extract.md` | `docs/` |
| `competition_tutorial.pdf` | `references/` |
| `submission_*.csv`, `catboost_info/`, `venv/` | removed from repo / gitignored |

> Do the moves with `git mv` **after** `git init` so history shows the reorganization as renames,
> not delete+add. The refactor of phase logic into `src/nfl_draft/` happens in Phase B (below);
> Phase A only relocates files.

---

## Phased plan (each phase independently valuable; ordered so you're never blocked)

### Phase A — Repo structure & hygiene (~half day)
Makes the project linkable/pushable immediately and gives it the industry layout; no ML changes.
- `git init` (from a copy or after moving `venv/` out); confirm the folder is clean.
- **Create the directory skeleton** from *Target repository structure* above and execute the
  **move mapping** with `git mv` (data → `data/raw/`, phase scripts → `experiments/`,
  notebooks → `notebooks/`, docs → `docs/`, pdf → `references/`). Add the empty `src/nfl_draft/`
  package tree with `__init__.py` files (bodies filled in Phase B).
- `.gitignore`: `venv/`, `catboost_info/`, `data/raw/*.csv` (confirm redistribution rights first),
  `data/processed/`, `submission_*.csv`, `models/*.joblib`, `models/*.cbm`, `wandb/`.
- **`pyproject.toml`** — package metadata (`nfl_draft`, `src/` layout, `pip install -e .`) + pinned
  deps; **`requirements.txt`** mirrors it. Verified versions: `lightgbm==4.6.0`, `catboost==1.2.10`,
  `optuna==4.9.0`, `scikit-learn==1.9.0`, `pandas==3.0.3`, `numpy==2.4.6`, `scipy==1.17.1`,
  plus new: `shap`, `wandb`, `modal`, `gradio`, `joblib`, `requests`.
- `README.md` (CV-ready): problem, the missingness insight, approach, **results table**,
  how-to-run, a directory-structure blurb, and a
  **`<!-- LEADERBOARD: rank pending ~Jul 2026 -->`** placeholder line to fill in.
- Reuse the existing strong walkthrough `docs/explanations/phase0_explanations.md` — link it from README.

### Phase B — Refactor to package + persist a deployable model (~1 day) ⭐ core new step
This is where the phase logic becomes the reusable `src/nfl_draft/` package **and** produces the
served artifact. Refactor, don't rewrite: lift the feature-engineering block from
`phase3_final.py:39-117` into `src/nfl_draft/features/build.py` (functions: `add_missingness`,
`add_composites`, `add_position_z`, `add_frequency`, `smooth_encode`) and the data loads into
`src/nfl_draft/data/load.py`, with all constants in `src/nfl_draft/config.py`.
- `src/nfl_draft/models/train.py` — instead of the CV/seed loop:
  - Fit `LGBMClassifier` on full train (with full-train `School_enc` via `smooth_encode`).
  - Fit `CatBoostClassifier` on full train (raw string `School`, `cat_feats`).
  - Save `lgbm_full.joblib`, `catboost_full.cbm`, `preprocess.joblib` into `models/`.
- `src/nfl_draft/models/predict.py` — `predict_one(player: dict) -> {prob, top_factors}`:
  single-row → `features.build` (same functions as training) → apply saved encoders → `0.85*lgbm + 0.15*cb`.
  Imported by both Modal and Gradio so the pipeline lives in exactly one place.
- Sanity assert: re-scoring `data/raw/train.csv` through `predict_one` roughly matches training AUC.
- The `experiments/phase*.py` scripts stay untouched — they are the historical record, not imports.

### Phase C — Tracking (W&B) + Explainability (SHAP) (~1 day)
- `src/nfl_draft/tracking/experiments.py`: a W&B run that logs the **phase progression**
  (0.8213 → 0.8276 → 0.8292 → 0.829, `experiments/phase3_final.py:223-227`), re-runs the Optuna study
  from `experiments/phase2_ensemble.py:133-150` logging all 50 trials, and uploads the model
  artifacts to W&B. Make the W&B project **public** for the CV link.
- SHAP: `shap.TreeExplainer` on `lgbm_full`; save a global feature-importance bar to
  `reports/figures/shap_importance.png` (expected top factors: `Age_missing`, `School_enc`,
  `num_drills_missing`) — this is the portfolio screenshot. Expose per-prediction top-3 SHAP
  factors from `predict.py` for the UI.

### Phase D — Modal serverless endpoint (~1 day) ⭐ new platform
- `app/modal_app.py`: `modal.Image` with the pinned deps, mount `models/`, expose a
  `@modal.fastapi_endpoint` (POST `/predict`) that accepts a player JSON and returns
  `{probability, top_factors}` via `predict_one`. `modal deploy modal_app.py` → public URL.
- Note in README that cold-start loads the joblib artifacts once per container.

### Phase E — Gradio UI (~half day)
- `app/gradio_app.py` (Gradio): form for the ~10 raw fields — `Age`, `School` (dropdown from train
  schools), `Height`, `Weight`, the 6 drill columns (blank-allowed → NaN), `Player_Type`,
  `Position_Type`, `Position`. On submit, POST to the Modal endpoint; render the draft
  probability + a small SHAP top-factors bar. Deploy on HF Spaces *as a thin client to Modal*
  (the Modal API is the showpiece, not the Gradio host).

---

## Deliverable files

See **Target repository structure** above for the full tree and which phase produces each file.
Mapping of deliverables → phase:
- Phase A: `README.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`, the directory skeleton + file moves.
- Phase B: `src/nfl_draft/{config.py, data/load.py, features/build.py, models/train.py, models/predict.py}`, `models/*` artifacts.
- Phase C: `src/nfl_draft/tracking/experiments.py`, `reports/figures/shap_importance.png`.
- Phase D: `app/modal_app.py`.
- Phase E: `app/gradio_app.py`.
- `tests/` (`test_features.py`, `test_predict.py`) alongside B/C.

`experiments/phase0..3_*.py`, `notebooks/`, and `docs/explanations/` stay as the
"how it was built" history.

---

## Verification (end-to-end, when implemented)

1. `pip install -e .` succeeds; `python -c "import nfl_draft"` imports cleanly.
2. `python -m nfl_draft.models.train` → `models/` populated; sanity AUC assert passes.
3. `python -c "from nfl_draft.models.predict import predict_one; print(predict_one({...}))"` → prob in [0,1];
   a player with all drills blank scores lower than a full-combine player (missingness signal works).
4. `pytest` → `test_features.py` + `test_predict.py` pass.
5. `python -m nfl_draft.tracking.experiments` → public W&B run shows 50 trials + phase table; `reports/figures/shap_importance.png` written.
6. `modal serve app/modal_app.py` then `curl -X POST .../predict -d '{...}'` → JSON prob + top_factors.
7. `python app/gradio_app.py` (or HF Space) → form submit hits Modal, shows prob + SHAP bar.

---

## CV / portfolio payoff

- **Card:** "NFL Draft Prediction — Gradient-Boosting Ensemble (0.83 AUC), served on Modal."
- **DS bullet (fill rank when out):** *Built and **deployed** a LightGBM+CatBoost ensemble
  (Optuna-tuned, 5-fold CV, seed-averaged) at 0.829 OOF AUC [**— top X%, GCI World 2026**];
  served it as a Modal serverless endpoint with a Gradio UI, W&B experiment tracking, and SHAP
  explainability, with leak-safe fold-internal target encoding.*
- Introduces **Modal + W&B + SHAP + model-serving** — platforms/skills absent elsewhere in the
  portfolio, directly addressing the Streamlit-monoculture problem.
