# Phase D · Step 3 — Permanent deploy + README

## 1. Big Picture
Step 2 proved the endpoint works, but a `modal serve` URL is temporary — it dies when you close the
terminal. Step 3 makes it **permanent**: `modal deploy` publishes a stable URL that keeps responding
with your laptop off, because the code and artifacts now live in Modal's cloud. Then we document it in
the README (including the serverless *cold-start* behaviour a reviewer should understand) so the live
API is a first-class part of the project story.

## 2. Core concepts, explained simply
- **`modal serve` vs `modal deploy`** — `serve` is dev mode: temporary URL (`…-predict-dev.modal.run`),
  tied to your terminal, hot-reloads. `deploy` is production: a **permanent** URL
  (`…-predict.modal.run`, no `-dev`), detached from your machine, updated only when you deploy again.
- **Cold start** — because Modal runs nothing while idle, the *first* request after a quiet period has
  to boot a fresh container and load the three artifacts (`lgbm_full.joblib`, `catboost_full.cbm`,
  `preprocess.joblib`) from disk — a few seconds. The container then stays **warm** and reuses the
  loaded models, so follow-up requests are fast. This is exactly why `predict.py` caches artifacts in
  module-level globals (`_ARTIFACTS`, `_EXPLAINER`): load once per container, reuse across requests.
- **Idempotent redeploys** — deploying again to the same app name **replaces** the old version at the
  same URL (no new URL, no leftover copies). Safe to run repeatedly.

## 3. File-by-file
- `README.md` — **edited.** Added a **Deployment — live prediction API (Modal)** section: the public
  URL, a copy-paste `curl` example, the blank-field/missingness note, and a **cold-start** callout.
  Ticked Phases B, C, D in the roadmap and updated the `app/` line in the structure blurb.
- `docs/explanations/phase_d_step{1,2,3}_explanation.md` — **new.** These write-ups.
- *(no source changes this step — deployment is a CLI action, not code.)*

Command run: `modal deploy app/modal_app.py` → deployed in ~4s (image already cached from Step 2) →
permanent URL `https://akhillavudya4567--nfl-draft-prediction-predict.modal.run`.

**Verification:** curling the permanent URL returned the same results as local/serve — full combine
**0.856**, all-drills-blank **0.071**, with `Age: null` + `Age_missing: 1` explaining the drop. The
deployed model matches local inference exactly.

## 4. Issues hit while building
- None new. (The `charmap`/UTF-8 fix from Step 2 was reused for `modal deploy`, since it prints the same
  `✓` output.)

## 5. Where things stand + what's next
**Phase D is complete.** The ensemble is a live, permanent, serverless API on Modal that returns a draft
probability + SHAP top factors, with the missing-data edge intact over HTTP. **Next (Phase E):** a small
**Gradio UI** — a form for the ~10 raw fields that POSTs to this Modal endpoint and renders the
probability + a SHAP top-factors bar, deployed as a thin client (Modal stays the showpiece).
