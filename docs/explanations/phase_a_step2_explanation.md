# Phase A · Step 2 — `pyproject.toml` + editable install

## 1. Big Picture
In Step 1 we could only import our package by hacking `sys.path` at the top of every script. That's
fragile and unprofessional. Step 2 fixes it permanently by **installing our own package into the
venv**, so `import nfl_draft` just works everywhere.

## 2. Core concepts, explained simply
- **`pyproject.toml`** — the modern standard file that describes a Python package: its name, its
  dependencies, and where its source code lives. Tools like `pip` read it to know how to build/install
  the project.
- **Editable install (`pip install -e .`)** — instead of *copying* your package's files into the venv
  (what a normal install does), it drops a *link* pointing back at your `src/` folder. Result: (1)
  `import nfl_draft` resolves from any directory, and (2) edits to your source take effect immediately,
  with no reinstall. The `.` means "install the package described by the `pyproject.toml` in this folder."
- **`src` layout** — keeping importable code under `src/` (not at the repo root) is an industry
  convention that prevents accidentally importing half-built code and forces you to test against the
  *installed* package. We tell setuptools about it with `[tool.setuptools.packages.find] where = ["src"]`.
- **Pinned dependencies** — writing `pandas==3.0.3` (exact version) instead of just `pandas` means the
  project rebuilds identically on any machine. We only pin what we actually use; later deps (`shap`,
  `wandb`, `modal`, `gradio`) get added when their phase arrives.

## 3. File-by-file
- `pyproject.toml` — **new, project root.** Declares the `nfl-draft-prediction` package, pins the 8
  core dependencies (versions read from the live venv), and points setuptools at `src/`.

## 4. Issues hit while building
- None. Verification ran the Step 1 import check **without** the `sys.path` hack and it still worked —
  proof the editable install is doing its job.

## 5. Where things stand + what's next
`nfl_draft` is now a properly installed, importable package with pinned deps. **Next (Step 3):**
`.gitignore` + `requirements.txt` — keep large/generated files (`venv/`, `data/raw/*.csv`, `models/`,
`catboost_info/`) out of version control, and provide a plain pinned dependency list for quick installs.
