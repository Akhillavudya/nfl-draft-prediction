# Phase D · Step 2 — The endpoint (`app/modal_app.py`) + `modal serve`

## 1. Big Picture
This is the heart of Phase D: one file that turns `predict_one` into a public web API. We describe the
container the model needs (an **Image**), copy the package and the saved artifacts into it, and expose a
`POST /predict` URL. Then we test it *before* committing to a permanent deploy, using `modal serve` — a
hot-reloading dev mode that gives a temporary URL. The whole point is that the served pipeline is
**byte-for-byte the same code** as local inference: the endpoint just calls `predict_one`, so there's no
training-vs-serving drift.

## 2. Core concepts, explained simply
- **`modal.Image`** — a recipe for the container's filesystem, built in layers: start from a base OS,
  `pip_install` the exact serving stack, set env vars, then copy local files in. Modal builds it once in
  the cloud and **caches** it, so later deploys are fast.
- **Why match versions & Python 3.11** — the saved model is a *pickle* (via joblib). Pickles are
  sensitive to the versions that created them, so the image pins the same `scikit-learn`, `numpy`,
  `pandas`, `lightgbm`, … and `python_version="3.11"` (your local Python). Mismatches can fail to
  unpickle or silently misbehave.
- **The dependency chain the image must satisfy** — `predict_one` imports `explain.top_factors`, which
  imports `train.py` and builds a SHAP explainer, and `explain.py` imports `matplotlib` at the top. So
  even though serving never draws a chart, the container still needs `shap` **and** `matplotlib`. Lesson:
  a container needs everything on the *import path*, not just what the request touches.
- **`add_local_dir` vs `add_local_python_source`** — two ways to get local files into the image:
  - `add_local_dir("models", "/models")` copies the artifacts folder to a fixed path `/models`.
  - `add_local_python_source("nfl_draft")` ships the package source and makes `import nfl_draft` work in
    the container — Modal handles *where* it lands and the import path for you.
  - Both are "local" layers and must come **last** in the recipe (nothing can build on top of them).
- **`fastapi[standard]`** — Modal's web endpoints are built on FastAPI, so it must be in the image. A
  function typed `def predict(player: dict)` makes FastAPI read the request's JSON body into `player`.
- **`@modal.fastapi_endpoint(method="POST", docs=True)`** — turns the function into an HTTP endpoint;
  `docs=True` also serves an interactive Swagger tester at `<url>/docs`.
- **Config override via env var** — `config.py` normally computes `MODELS_DIR` from the repo's folder
  layout, which doesn't exist inside a container. We made it read `NFL_DRAFT_MODELS_DIR` if set (the
  image sets it to `/models`), else fall back to the old path. This is the **12-factor** idea: don't
  hard-code deployment paths in code; let the environment override them. Training/tests are unaffected
  because the default is unchanged.
- **Imports inside the function** — `from nfl_draft.models.predict import predict_one` sits *inside*
  `predict`, not at the top of the file. When you run `modal serve`, this file executes on your laptop
  as the "driver" that defines the app; the heavy ML imports belong in the **container**, so they live
  in the function body and only run there.
- **`modal serve`** — dev mode: deploys to the cloud but ties the app to your terminal and hot-reloads
  on file edits, giving a temporary `…-dev.modal.run` URL. Great for testing before a real deploy.

## 3. File-by-file
- `app/modal_app.py` — **new.** The entire deployment:
  - `app = modal.App("nfl-draft-prediction")` — the named handle.
  - `image = modal.Image.debian_slim(python_version="3.11").pip_install(…serving stack + fastapi…)
    .env({"NFL_DRAFT_MODELS_DIR": "/models"}).add_local_dir("models", "/models")
    .add_local_python_source("nfl_draft")` — the container recipe.
  - `predict(player: dict)` — decorated as a `POST` endpoint; imports and calls `predict_one(player)`.
- `src/nfl_draft/config.py` — **edited.** Added `import os` and made the models path overridable:
  `MODELS_DIR = Path(os.environ.get("NFL_DRAFT_MODELS_DIR", PROJECT_ROOT / "models"))`. Default
  behaviour (local training, tests) is identical to before.

**What the test showed** (`modal serve`, then curl the `-dev` URL): a full-combine WR scored **0.856**;
the same player with every drill omitted scored **0.071**, and `top_factors` reported `Age: null` and
`Age_missing: 1` as the reasons. The missing-data edge survives the HTTP round trip — a blank field
stayed `NaN`, never `0`.

## 4. Issues hit while building
- **`'charmap' codec can't encode character '✓'`** (Modal crashed immediately on `serve`).
  - *What happened:* Modal's CLI prints a `✓`. We redirected its output to a log file; once output isn't
    a terminal, Python falls back to Windows' legacy `cp1252` ("charmap") encoding, which has no `✓`.
  - *Why:* on Windows, stdout defaults to the locale codec, not UTF-8, when piped/redirected.
  - *Fix:* set `PYTHONIOENCODING=utf-8` (we also set `PYTHONUTF8=1`) before the command.
  - *Recognise next time:* any `UnicodeEncodeError` / `charmap … can't encode` from a CLI you're piping
    or redirecting on Windows → force UTF-8 with `PYTHONIOENCODING=utf-8`.
- **First build is slow, later ones aren't.** *Lesson:* the image (installing the full ML stack in Linux)
  builds once in the cloud, then Modal caches it — expect a couple of minutes the first time only.

## 5. Where things stand + what's next
`app/modal_app.py` serves `predict_one` over HTTP, verified end-to-end via `modal serve`, and
`config.py` is deployment-aware. **Next (Step 3):** `modal deploy` for a **permanent** URL (works with
your laptop off), a curl to confirm, and the README cold-start note.
