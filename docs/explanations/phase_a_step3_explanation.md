# Phase A · Step 3 — Repo hygiene + first commit & push

## 1. Big Picture
Our code was on disk but not under version control and not backed up anywhere. This step makes the
project a real **git repository**, keeps junk out of it, and publishes it to **GitHub** so it's
shareable (and CV-visible).

## 2. Core concepts, explained simply
- **`.gitignore`** — a list of path patterns git should *never* track. We use it to exclude the huge
  `venv/`, the data CSVs, generated model artifacts, `catboost_info/`, and secrets. **Why it matters:**
  anything committed once lives in git history *forever* — so we exclude it *before* the first commit,
  not after.
- **`.gitkeep`** — git can't track an *empty* folder. To keep structural folders like `models/` and
  `reports/figures/` in the repo, we drop an empty `.gitkeep` file inside and un-ignore it.
- **`requirements.txt`** — a plain pinned dependency list. We keep it *and* `pyproject.toml`:
  `pyproject.toml` defines the installable package; `requirements.txt` is a quick "install these exact
  versions" convenience list.
- **Repository / remote / origin** — the project now lives in two places: locally, and on GitHub. A
  **remote** is a named link to a hosted copy; **`origin`** is the conventional name for the main GitHub
  copy. `git push` uploads local commits to it.
- **Editable install note:** because our package is installed with `pip install -e .`, git also ignores
  the generated `*.egg-info/` metadata.

## 3. File-by-file
- `.gitignore` — **new.** Excludes `venv/`, `data/raw/*.csv`, `data/processed/*`, `models/*`,
  `catboost_info/`, `submission_*`, `reports/figures/*.png`, `wandb/`, `.env`, and Python cruft; keeps
  the `.gitkeep` placeholders.
- `requirements.txt` — **new.** The 8 pinned core deps, mirroring `pyproject.toml`.
- `data/raw/.gitkeep`, `data/processed/.gitkeep`, `models/.gitkeep`, `reports/figures/.gitkeep` —
  **new, empty.** Preserve otherwise-empty folders in the repo.

## 4. Issues hit while building
- **LF → CRLF warnings on commit.** *What:* git warned it would convert line endings on many files.
  *Why:* Windows uses CRLF line endings, Unix/git prefer LF; git normalizes them. *Fix:* nothing needed
  — it's a harmless notice, not an error. *Lesson:* on Windows this is normal; you can silence it later
  with a `.gitattributes` file if desired.
- **Safety check before staging.** We ran `git check-ignore` and `git status` *before* `git add` to
  confirm `venv/` and the CSVs were excluded. *Lesson:* always look before you add — a committed secret
  or huge file is painful to remove from history.

## 5. Where things stand + what's next
The repo is initialized, cleanly committed (no AI co-author trailer), and pushed to GitHub on `main`.
**Next (Step 4, final Phase A step):** write `README.md` — the CV-facing front page: the problem, the
missingness insight, the results table, and how to run it.
