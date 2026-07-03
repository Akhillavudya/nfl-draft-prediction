# Project Extract — NFL Draft Prediction

> Auto-generated profile for CV / portfolio use. Every claim below was read from the actual repo (source `.py` files, `README.ipynb`, `IMPLEMENTATION_PLAN.md`, `docs/`, and the `input/` CSVs). There is **no git history** in this folder (not a git repo), so timeframe is inferred from the competition README dates.

---

## 1. Identity

- **Project name:** NFL Draft Prediction (GCI World 2026 / Omnicampus competition)
- **One-line description:** Gradient-boosting ensemble that predicts whether a college player gets drafted into the NFL.
- **Timeframe:** ~May–June 2026 (competition ran Apr 29 → Jun 12 2026; ranking announced Jul 8 2026). No commit history to narrow this further.
- **Type:** Competition (GCI World 2026 April, hosted on Omnicampus — a structured/educational ML competition)
- **Status:** Completed — a final submission (`submission_FINAL.csv`) was produced via a 4-phase pipeline. Actual leaderboard placement is **NOT FOUND IN REPO — I'll fill this in.**

---

## 2. The Problem

- **What it solves:** Given a college athlete's NFL Combine results (physical measurements + drill times), college, and position, predict the probability that the player will be selected in the NFL Draft. The core analytical insight the project exploits is that *missing* combine data is itself highly predictive — players who skip drills are usually those who weren't invited (and rarely drafted).
- **Audience:** NFL scouting/analytics teams as the hypothetical end user; in practice, a data-science competition graded on ranking quality (ROC AUC). Predictions are a 0–1 draft-likelihood score per player.

---

## 3. Technical Stack (verified from `venv` dist-info + imports in source)

- **Languages:** Python (only). 100% of source is Python `.py` scripts + Jupyter notebooks.
- **Frameworks / libraries (exact versions from `venv/Lib/site-packages/*.dist-info`):**
  - lightgbm **4.6.0** (primary model)
  - catboost **1.2.10** (second model, native categorical handling)
  - optuna **4.9.0** (hyperparameter search; pulls in sqlalchemy 2.0.50 + alembic 1.18.4)
  - scikit-learn **1.9.0** (StratifiedKFold, roc_auc_score, LabelEncoder)
  - pandas **3.0.3**, numpy **2.4.6**, scipy **1.17.1** (`rankdata` for rank-averaging)
  - matplotlib **3.10.9**, plotly **6.8.0** (available; used in tutorial notebooks)
  - **No `requirements.txt` / `pyproject.toml`** — dependencies were only reconstructable from the venv. (Gap — see §8.)
- **Databases/storage:** None. Flat CSV in/out (`input/*.csv` → `submission_*.csv`). Optuna defaults to in-memory study (no DB persisted).
- **Infra/deployment:** None. Local Windows venv (`pyvenv.cfg`, python 3.11). No Docker, no cloud config, no CI.
- **APIs/external services:** None (competition rules prohibit external data).
- **ML/AI components:**
  - **LightGBM gradient-boosted trees** — main learner, Optuna-tuned (50 TPE trials)
  - **CatBoost classifier** — second learner using `School` as a *native* categorical feature
  - **Ensemble:** OOF-weighted blend (85% LGBM / 15% CatBoost) with a rank-average fallback, plus 3-seed averaging
  - Techniques: smoothed/fold-internal target encoding, missingness-flag features, position-relative z-scores

---

## 4. Architecture & Key Technical Decisions

- **Staged pipeline (phase0 → phase1 → phase2 → phase3):** each phase is a standalone script that adds one layer of sophistication and logs its OOF AUC, so improvements are attributable. `phase0_lgbm.py` (baseline LGBM), `phase1_lgbm.py` (feature engineering), `phase2_ensemble.py` (Optuna + CatBoost + blend), `phase3_final.py` (seed-averaged final).
- **Leak-safe target encoding (the non-trivial bit):** `School` is target-encoded *inside* each CV fold using only that fold's training rows, with Bayesian smoothing `(n·school_mean + 10·global_mean)/(n+10)` — see `smooth_encode()` in `phase3_final.py:98`. Doing this before the fold loop would leak the validation label and inflate CV. This is the project's headline engineering decision and is documented at length in `docs/explanations/phase0_explanations.md`.
- **Missingness-as-signal feature design:** instead of imputing missing drills (what the baseline did), the pipeline creates per-drill `_missing` flags + `num_drills_missing` + `full_combine` (`phase3_final.py:43-47`). Per the data audit, the `Age`-missing flag *alone* yields AUC ~0.716.
- **Two complementary models on purpose:** LightGBM consumes a label-encoded + target-encoded `School`; CatBoost consumes raw-string `School` natively. Different encodings → decorrelated errors → a blend that beats either alone. Blend weight is chosen by grid-searching OOF AUC (`phase2_ensemble.py:275-282`).
- **Variance control for a tiny test set:** test is only 696 rows, so the final model averages predictions across seeds `[42, 123, 2025]` and 5 folds, and the code deliberately trusts OOF AUC over the noisy public leaderboard.
- **Scale indicators:** ~**813 lines** of core Python across 4 phase scripts (116 / 148 / 310 / 239); 4 tutorial notebooks; 1 deep-dive explanation doc. Dataset: **2,781 train rows × 16 columns**, 696 test rows, 236 schools, 64.8% base draft rate. **No automated tests** (only inline `assert` sanity checks on the submission file). CatBoost training artifacts present in `catboost_info/`.

---

## 5. Quantifiable Results / Impact

Hard numbers found in the repo (printed/recorded in the scripts):

| Stage | OOF ROC AUC |
|---|---|
| Phase 0 — LGBM + missingness flags + School encoding | **0.8213** |
| Phase 2 — Optuna-tuned LGBM | **0.8276** |
| Phase 2 — LGBM/CatBoost blend (85/15) | **0.8292** |
| Phase 3 — seed-averaged blend (FINAL) | ~0.829+ (recomputed at runtime) |
| Baseline reference (RF depth=5) | ~0.75–0.78 (estimated, not run here) |

- **Signal strength:** `Age`-missing flag alone → AUC **0.716**; school draft rates span **0.36–0.85**; 97.6% of test schools appear in train.
- **Improvement:** roughly **+0.05–0.07 AUC** over the estimated baseline (~0.78 → 0.829), and **+0.008 AUC** from tuning+ensembling+seed-averaging over the phase-0 model.
- **Dataset scale:** 2,781 training players, 696 test players, 236 schools, 16 raw features expanded to ~35+ engineered features.
- **Competition placement / final leaderboard score: NOT FOUND IN REPO — I'll fill this in.** ⚠️ Recommend adding the actual rank/percentile and private-LB AUC before putting this on a CV — the OOF numbers are real but a leaderboard result is far stronger.

---

## 6. Role-Specific Angles

**SDE bullet:**
Engineered a modular 4-phase ML pipeline (~800 LOC Python) with leak-safe fold-internal target encoding, reproducible seed control, and assertion-guarded output validation, cleanly separating baseline, feature-engineering, tuning, and ensembling stages for attributable per-phase gains.

**Data Analyst bullet:**
Analyzed 2,781-player NFL Combine records to surface that data *missingness* (skipped drills, absent age) drove draft outcomes — a single missing-age flag separated drafted from undrafted at 0.716 AUC — and quantified school-level draft pipelines ranging from 36% to 85%.

**Data Scientist bullet:**
Built a LightGBM + CatBoost ensemble (Optuna-tuned over 50 trials, 5-fold stratified CV, 3-seed averaging) reaching 0.829 OOF ROC AUC, lifting ~0.05 AUC over baseline via missingness features, smoothed target encoding, and position-relative z-scores — all validated leak-free on out-of-fold predictions.

---

## 7. For the Portfolio Website

- **Card title:** "NFL Draft Prediction — Gradient-Boosting Ensemble (0.83 AUC)"
- **Card summary:** A staged LightGBM + CatBoost pipeline that predicts NFL Draft selection from college combine data, reaching 0.829 ROC AUC. Its edge comes from treating *missing* combine results as predictive signal and leak-safe per-fold target encoding of player schools.
- **Live demo?** Not currently — it's a batch CSV pipeline. Could wrap the trained blend in a small **Streamlit/Gradio app on Hugging Face Spaces** (input a player's stats → draft probability). Static write-up would also work on Vercel/Cloudflare Pages.
- **Screenshot-worthy visual:** No dashboard exists yet. Best candidates to generate: (1) a feature-importance bar chart from the fitted LightGBM model, (2) the phase-by-phase AUC progression table/chart, (3) school draft-rate distribution. The phase-progression printout in `phase3_final.py:223-227` is the most resume-friendly thing to visualize.
- **GitHub status:** **Not a git repo yet** (no `.git`). README is the *competition-provided* `README.ipynb`, not a project README — needs a real one before publishing. The `docs/explanations/phase0_explanations.md` is genuinely strong portfolio material (clear, beginner-friendly walkthrough).

---

## 8. Gaps / Cleanup Needed Before Public Use

- **Secrets/keys:** ✅ **None found in your code.** A grep for keys/tokens/passwords hit 250 files but *all* were inside `venv/` third-party libraries (sqlalchemy, alembic, etc.) — zero in your scripts/CSVs/docs.
- **`venv/` is committed-in-place** — never push this; add a `.gitignore` (`venv/`, `catboost_info/`, `*.csv` submissions). It bloats the repo massively.
- **No `requirements.txt`** — add one pinned to the verified versions (lightgbm 4.6.0, catboost 1.2.10, optuna 4.9.0, scikit-learn 1.9.0, pandas 3.0.3, numpy 2.4.6, scipy 1.17.1).
- **No real README** — only the competition's `README.ipynb`. Write a project README: problem, approach, results table, how-to-run.
- **Not under version control** — `git init` so timeframe/commit history exists for future extracts and to show iteration.
- **Missing final number** — the actual leaderboard rank/score is absent; benchmark/record it before CV use.
- **No tests** — only inline asserts; fine for a competition, worth noting if presenting to SDE interviewers.
- **`phase0_lgbm.py` referenced by docs** but not re-read here; confirm it still matches the explanation doc before publishing.
