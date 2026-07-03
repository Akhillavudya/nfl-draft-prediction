<!-- LEADERBOARD: rank pending ~Jul 2026 — fill in final rank/percentile here when it lands -->

# NFL Draft Prediction 🏈

Predicting whether a college football player will be **selected in the NFL Draft** from their
Combine measurements, drill times, school, and position — a **LightGBM + CatBoost** gradient-boosting
ensemble reaching **0.829 out-of-fold ROC AUC**.

Built for the **GCI World 2026 / Omnicampus** competition (metric: ROC AUC).

![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-brightgreen)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)
![OOF AUC](https://img.shields.io/badge/OOF%20ROC%20AUC-0.829-success)

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

| Stage | OOF ROC AUC |
|---|---|
| Phase 0 — LightGBM + missingness flags + School encoding | 0.8213 |
| Phase 2 — Optuna-tuned LightGBM | 0.8276 |
| Phase 2 — LightGBM / CatBoost blend (85 / 15) | 0.8292 |
| **Phase 3 — seed-averaged blend (final)** | **~0.829** |

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

## Project structure

```
nfl-draft-prediction/
├── data/raw/                 # train.csv, test.csv, sample_submission.csv  (not committed)
├── src/nfl_draft/            # reusable package — the maintained code path
│   ├── config.py             # all paths, seeds, params, column lists
│   ├── data/                 # data loading
│   ├── features/             # feature engineering (the shared pipeline)
│   ├── models/               # training + inference
│   └── tracking/             # experiment tracking + explainability
├── experiments/              # original staged scripts (phase0→phase3) — how it was built
├── notebooks/                # EDA + tutorials
├── app/                      # serving endpoint + UI  (in progress)
├── models/                   # saved model artifacts  (generated, not committed)
├── reports/figures/          # charts, e.g. SHAP importance
├── docs/                     # plan + per-step beginner explanations
├── pyproject.toml            # installable package + pinned deps
└── requirements.txt
```

The `experiments/phase0..3_*.py` scripts are preserved as the build history (each adds one layer and
logs its AUC). The maintained, reusable pipeline is being consolidated into `src/nfl_draft/`.

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

# 4. Add the competition data (not redistributed here) into:
#    data/raw/train.csv  data/raw/test.csv  data/raw/sample_submission.csv
```

> The dataset is not included in this repo (competition data). Download it from the competition page and
> place the three CSVs in `data/raw/`.

---

## Roadmap

This repo is being extended from a batch competition script into a **shipped, tracked, explainable**
model. See [`docs/final-plan.md`](docs/final-plan.md).

- [x] **Phase A** — industry project structure, installable `nfl_draft` package, repo hygiene
- [ ] **Phase B** — persist a full-train model (`joblib`) + single-row inference
- [ ] **Phase C** — Weights & Biases experiment tracking + SHAP explainability
- [ ] **Phase D** — Modal serverless prediction endpoint
- [ ] **Phase E** — Gradio UI

Beginner-friendly write-ups of each step live in [`docs/explanations/`](docs/explanations/).

---

## Tech stack

Python · LightGBM · CatBoost · Optuna · scikit-learn · pandas / NumPy / SciPy

## License

Released under the MIT License (add a `LICENSE` file to formalize).
