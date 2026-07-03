# Phase A · Step 1 — Make `nfl_draft` an importable package

## 1. Big Picture
We're turning a pile of flat scripts into an **industry-standard project layout**. The very first
brick is turning `src/nfl_draft/` from "just some folders" into a real Python **package** — something
other files can `import` from. Without this, every script has to copy-paste the same constants
(paths, seeds, model settings). With it, there's **one** shared source of truth.

## 2. Core concepts, explained simply
- **Package** — a folder Python is allowed to `import` from. A folder becomes a package only when it
  contains a file literally named `__init__.py`. That file can be completely empty; its *presence* is
  the signal. We added one to `nfl_draft/` and each sub-folder (`data/`, `features/`, `models/`,
  `tracking/`).
- **`config.py`** — one module holding every constant the project needs. Instead of `SEED = 42`
  appearing in four different scripts, it lives here once and everyone does `from nfl_draft import config`.
- **`Path(__file__).resolve().parents[2]`** — `__file__` is *this file's own location on disk*.
  `.parents[2]` walks up two folders (`config.py` → `nfl_draft/` → `src/` → project root). This makes
  paths like `data/raw/train.csv` resolve correctly **no matter which folder you run from**, with no
  brittle hard-coded `C:/Users/...` path.

## 3. File-by-file
- `src/nfl_draft/__init__.py` (+ the four sub-folder `__init__.py`s) — **new, empty.** Mark the folders
  as packages.
- `src/nfl_draft/config.py` — **new.** Holds paths, `SEED`/`SEEDS`/`N_FOLDS`, the column lists
  (`DRILL_COLS`, `DRILL_VALS`, `CAT_COLS`, `CAT_FEATS`, `DROP_COLS`), `LGBM_PARAMS`, `CATBOOST_PARAMS`,
  and the blend weights (`W_LGBM`/`W_CB`, `SMOOTH`). These values were lifted verbatim from
  `experiments/phase3_final.py`.

## 4. Issues hit while building
- None this step. Verified with a one-liner that imported `nfl_draft.config` and checked
  `TRAIN_CSV.exists()` → `True`, confirming both the import and the moved data path work.

## 5. Where things stand + what's next
The package is importable and its config resolves the moved `data/raw/` files. **Next (Step 2):** add
`pyproject.toml` so `import nfl_draft` works everywhere via an *editable install* (`pip install -e .`) —
removing the temporary `sys.path` hack used in the verification command.
