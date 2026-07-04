# NFL Draft Prediction 🏈

Predicting whether a college football player will be **selected in the NFL Draft** from their Combine
measurements, drill times, school, and position — a **LightGBM + CatBoost** gradient-boosting ensemble
reaching **0.829 out-of-fold ROC AUC**, served as a live serverless API with a web UI, experiment
tracking, and per-prediction explanations.

Built for the **GCI World 2026 / Omnicampus** competition (metric: ROC AUC).

![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-brightgreen)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)
![OOF AUC](https://img.shields.io/badge/OOF%20ROC%20AUC-0.829-success)
![Modal](https://img.shields.io/badge/deployed-Modal-7c3aed)

---

## Highlights

- **0.829 out-of-fold ROC AUC** from a LightGBM + CatBoost ensemble (Optuna-tuned, 5-fold CV, seed-averaged).
- **Missing data treated as signal** — the core edge; a single missing-`Age` flag scores ≈0.716 AUC alone.
- **Leak-safe target encoding** of `School`, computed inside each CV fold with Bayesian smoothing.
- **Live prediction API** on Modal (serverless) — POST a player's stats, get a probability back.
- **Web UI** (Gradio) as a thin client to the API.
- **Explainable** — SHAP global importances plus top factors behind every individual prediction.
- **Reproducible** — installable `src/` package, pinned dependencies, and tests.

---

## The interesting bit — *missing data is the signal*

The naive approach fills in missing Combine results with averages. This project does the opposite: it
treats **the absence of a measurement as information**, because players who skip drills are usually the
ones who weren't invited — and rarely get drafted.

| Insight | Evidence |
|---|---|
| A **missing `Age`** flag, *by itself*, separates drafted from undrafted | AUC ≈ **0.716** from that one feature |
| More missing drills → lower draft rate | 0 missing → 78.5% drafted · 7 missing → ~0% |
| **School** matters, and generalizes | draft rates span 0.36–0.85; 97.6% of test schools seen in training |

The second non-trivial piece is **leak-safe target encoding** of `School`: the school→draft-rate map is
computed *inside each cross-validation fold* (with Bayesian smoothing), so a player's own outcome never
leaks into their own feature. Doing this naively inflates your validation score and collapses on the
real leaderboard.

---

## Results

| Model | OOF ROC AUC |
|---|---|
| LightGBM + missingness flags + School encoding | 0.8213 |
| &nbsp;&nbsp;+ Optuna-tuned hyperparameters | 0.8276 |
| &nbsp;&nbsp;+ CatBoost blend (85 / 15) | 0.8292 |
| &nbsp;&nbsp;+ seed-averaging (**final**) | **~0.829** |

**Competition placement:** _pending — the leaderboard is released ~July 2026._

Scores are **out-of-fold** (5-fold stratified CV, averaged over 3 seeds) — the model never sees the rows
it's scored on, so these numbers reflect true generalization, not memorization.

---

## Approach

- **Two complementary models on purpose.** LightGBM consumes a label- and target-encoded `School`;
  CatBoost consumes the raw `School` string natively. Different encodings → decorrelated errors → a
  blend that beats either model alone.
- **Feature engineering:** per-drill `_missing` flags, `num_drills_missing`, physical composites
  (BMI, power, speed score, agility diff), and **position-relative z-scores** (how a player compares to
  others at their position).
- **Tuning:** 50-trial Optuna search on the LightGBM hyperparameters.
- **Variance control:** the test set is only 696 rows, so final predictions are averaged across seeds
  `[42, 123, 2025]` — and the model trusts out-of-fold AUC over the noisy public leaderboard.

---

## Getting started

```bash
# 1. Clone
git clone https://github.com/Akhillavudya/nfl-draft-prediction.git
cd nfl-draft-prediction

# 2. Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Install the package (editable) with pinned dependencies
pip install -e .

# 4. Add the competition data (not redistributed here) into data/raw/:
#    train.csv  test.csv  sample_submission.csv
```

> The dataset is not included in this repo (competition data). Download it from the competition page and
> place the three CSVs in `data/raw/`.

**Train and persist the model, then predict a single player:**

```bash
# Fit on the full training set and save the model + preprocessing artifacts into models/
python -m nfl_draft.models.train

# Score one player from Python
python -c "from nfl_draft.models.predict import predict_one; \
print(predict_one({'Age':22,'Height':74,'Weight':210,'School':'Alabama', \
'Player_Type':'offense','Position_Type':'backs_receivers','Position':'WR'}))"
```

Run the tests with `pytest`.

---

## Live prediction API (Modal)

The trained ensemble is served as a **serverless endpoint on [Modal](https://modal.com)**. POST a
player's raw stats as JSON; get back a draft probability plus the top SHAP factors behind it.

```bash
curl -X POST https://akhillavudya4567--nfl-draft-prediction-predict.modal.run \
  -H "Content-Type: application/json" \
  -d '{"Age":22,"Height":74,"Weight":210,"Sprint_40yd":4.5,"Vertical_Jump":35,
       "Bench_Press_Reps":18,"Broad_Jump":120,"Agility_3cone":6.9,"Shuttle":4.2,
       "School":"Alabama","Player_Type":"offense","Position_Type":"backs_receivers","Position":"WR"}'
# -> {"probability": 0.856, "top_factors": [...]}
```

**Leave a Combine field blank** (omit it) and it stays `NaN`, never `0` — so the `_missing` flags fire
and the probability drops sharply (the same player with all drills blank scores ~0.07). The model's
missing-data edge holds end-to-end over HTTP.

> **Cold start:** Modal runs *no* server while idle (and bills nothing). The first request after a quiet
> period spins up a fresh container and loads the model artifacts once — a few seconds of latency. The
> container then stays warm and reuses the loaded models, so follow-up calls are fast.

Deploy your own copy from the repo root with `modal deploy app/modal_app.py` (after `modal setup`).

---

## Web UI (Gradio)

`app/gradio_app.py` is a small **[Gradio](https://gradio.app) form** — a *thin client* that does no ML
itself. It collects a player's raw stats, POSTs them to the Modal endpoint, and renders the draft
probability plus the top SHAP factors. Clearing a Combine field leaves it blank (`NaN`), so the
missing-data signal drives the result just as it does over `curl`.

```bash
python app/gradio_app.py          # serves a local URL (defaults to the deployed Modal endpoint)
```

Set `MODAL_ENDPOINT_URL` before launching to point it at your own deploy. Because it's a thin client, it
can be hosted anywhere (e.g. Hugging Face Spaces) while Modal remains the model service.

---

## Explainability (SHAP)

A [SHAP](https://github.com/shap/shap) `TreeExplainer` over the LightGBM model produces both a global
feature-importance chart (`reports/figures/shap_importance.png`) and the top-3 factors behind each
individual prediction, which the API and UI surface alongside the probability. Experiment runs —
hyperparameter trials and the ensemble progression — are tracked in Weights & Biases.

---

## Project structure

```
nfl-draft-prediction/
├── data/raw/                 # train.csv, test.csv, sample_submission.csv  (not committed)
├── src/nfl_draft/            # reusable package — the maintained code path
│   ├── config.py             # all paths, seeds, params, column lists
│   ├── data/                 # data loading
│   ├── features/             # feature engineering (the shared pipeline)
│   ├── models/               # training, inference, and SHAP explanations
│   └── tracking/             # experiment tracking (Weights & Biases)
├── app/                      # modal_app.py (serving endpoint) + gradio_app.py (UI)
├── experiments/              # original staged scripts — the iterative build history
├── notebooks/                # EDA + tutorials
├── models/                   # saved model artifacts  (generated, not committed)
├── reports/figures/          # charts, e.g. SHAP importance
├── docs/                     # design docs + explanations
├── tests/                    # feature + prediction tests
├── pyproject.toml            # installable package + pinned deps
└── requirements.txt
```

A single shared feature pipeline in `src/nfl_draft/features/` is used by both training and serving, so
the model can't drift between the two.

---

## Tech stack

Python · LightGBM · CatBoost · Optuna · scikit-learn · SHAP · Weights & Biases · Modal · Gradio ·
pandas / NumPy / SciPy

## License

Released under the MIT License (add a `LICENSE` file to formalize).
