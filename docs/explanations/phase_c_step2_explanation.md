# Phase C · Step 2 — Experiment tracking with Weights & Biases

## 1. Big Picture
Step 1 made the model *explainable*. Step 2 makes the whole modelling process **legible and
shareable**. Right now the story of *how* the model got to 0.829 — the per-phase AUC gains, the 50
hyperparameter trials — lives only in the terminal output of scripts that already ran. **Weights &
Biases (W&B)** is a hosted logbook: we re-run the tuning, log everything to a **public dashboard**, and
attach the trained model files. The payoff is a single URL a recruiter can open to see the work, and a
concrete new skill ("experiment tracking") the portfolio was missing.

## 2. Core concepts, explained simply
- **Experiment tracking** — a logbook-as-a-service. You call `wandb.log({...})` during a run and it
  stores metrics, tables, images, and files, then renders them on a web dashboard you can share.
- **Run** — one tracked execution (`wandb.init(...)` starts it, `run.finish()` ends it). Everything you
  log belongs to that run and gets a URL.
- **Artifact** — a **versioned** bundle of files attached to a run. We upload the three model files so
  the exact model behind a result is downloadable and reproducible, with version history.
- **`.env` + `python-dotenv`** — the W&B API key is a secret, so it lives in `.env` (git-ignored).
  `load_dotenv()` reads it into an environment variable at runtime, so the key never touches code or git
  history. `.env.example` (committed) documents the *name* of the variable only, never the value.
- **Optuna study / trial** — Optuna searches for good hyperparameters by trying many configurations
  (each try = a **trial**) and steering toward better ones (a **study** = the whole search). We log each
  trial's AUC so the dashboard shows the optimisation curve.
- **Leak-safe target encoding, per fold** — the CV re-computes the `School`→draft-rate map *inside each
  training fold* and applies it to the held-out fold. Building that map from the whole dataset would
  leak the answer into validation and inflate the score. This is exactly why the tracking script builds
  the feature matrix **without** `School_enc` and injects it fold-by-fold.

## 3. File-by-file
- `src/nfl_draft/tracking/experiments.py` — **new.** The W&B run:
  - `_base_lgbm_matrix()` — label-encoded features **without** `School_enc` (added per fold).
  - `run_lgbm_cv()` / `_objective()` — the leak-safe 5-fold Optuna objective, lifted faithfully from
    `experiments/phase2_ensemble.py`.
  - `main()` — `wandb.init` → log the phase-progression table → run 50 Optuna trials (a callback logs
    each trial live) → regenerate + log the SHAP image → upload the 3 model artifacts → write the best
    params/AUC into the run summary.
- `.env.example` — **new.** Committed placeholder documenting `WANDB_API_KEY=`.
- `pyproject.toml` / `requirements.txt` — **edited.** Pinned `wandb==0.28.0` and `python-dotenv==1.2.2`.

**Result of the run:** best OOF AUC **0.8276** with best params (`lr 0.0645, num_leaves 41,
max_depth 8, min_child 100, feat_frac 0.557, bag_frac 0.924, alpha 0.0059, lambda 3.61`) — these
**exactly match** the frozen `LGBM_PARAMS` in `config.py`, which confirms the leak-safe CV was
reproduced faithfully (a built-in checksum, not a coincidence). Public run:
<https://wandb.ai/akhillavudya-iit-guwahati-spe-student-chapter/nfl-draft-prediction/runs/j2ap7f1v>

## 4. Issues hit while building
- **W&B API key looked wrong (86 chars).**
  - *What happened:* the loaded key was 86 characters; older W&B keys are ~40, so it looked like a bad
    paste.
  - *Why/fix:* newer W&B keys are simply longer and valid. We verified with
    `wandb.login(key=..., verify=True)` **before** the multi-minute run instead of guessing from length —
    and never printed the key.
  - *Lesson:* validate credentials against the server up front; judge a secret by whether it
    authenticates, not by how it looks.
- **Full-train vs. per-fold encoding (a design trap avoided).**
  - *What/why:* the reusable `train.py` bakes `School_enc` in using the **full-train** map — correct for
    the final served model, but it would **leak** inside CV. So the tracking script deliberately builds
    the matrix *without* `School_enc` and re-derives it per fold.
  - *Lesson:* the same feature can be computed differently for *serving* (full-train) vs. *evaluation*
    (per-fold). Keep the leak-safe version for any CV number you report.

## 5. Where things stand + what's next
Phase C is complete: the model is **explainable** (Step 1) and **tracked** (Step 2) on a public W&B
project showing the phase table, 50 Optuna trials, the SHAP figure, and the downloadable model
artifacts. Remaining one-time action: toggle the W&B **project to public** in its settings so the CV
link is world-viewable. **Next (Phase D):** wrap `predict_one` behind a Modal serverless POST
`/predict` endpoint — the JSON-safety fix from Step 1 is what makes that clean.
