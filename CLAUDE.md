# CLAUDE.md — How to work with me on this project

> This file is instructions for Claude Code (and any AI assistant) working in this repo. It is loaded
> automatically at the start of every session. Read it before doing anything else.

---

## Who I am (the human) and what I'm trying to get out of this

I am a **beginner** building this project to **learn software and AI engineering from scratch** — not
just to get a working app. The learning matters more than the speed. If you optimize for "task done
fast," you are failing me. Optimize for "I understand what happened and could redo it myself."

Concretely, I am converting a working competition script (a LightGBM + CatBoost NFL-Draft
prediction pipeline) into an industry-structured, CV-ready project, following `docs/final-plan.md`.
But the *point* is that I come out of it actually understanding Python packaging, ML pipelines,
model persistence, serverless deployment, experiment tracking, and explainability — well enough to
explain them and build the next one myself.

---

## The core rules (these override default behavior — follow them exactly)

### 1. I write the code, not you.
**Do not write or paste application code directly into files.** Instead:
- Give me the code **in the chat**, with the file path and a one-line purpose.
- Let me paste it into the file myself.
- I learn by typing/placing each piece — that's the whole point.

**This applies to:** all package/feature/app code (`src/nfl_draft/*`, `app/*`, the training/predict/
tracking modules, config modules, etc.).

**This does NOT apply to — you may edit these directly:**
- Pure repo-hygiene files: `.gitignore`, `README.md`, `.env.example`, this `CLAUDE.md`, docs.
- Fixing an obvious paste mistake I just made (a typo, a missing line I clearly intended).
- Verification / cleanup actions: deleting confirmed-dead files, running checks, cleaning up test data.
- **Never** put real secrets in a file that gets committed. Real keys go in `.env` (git-ignored) only.

**Comment the code you give me — but sparingly.** Give **one** short, one-line comment per function
(and per top-level module-level construct) saying what it does. Do **NOT** put a comment on every line
or after each block — that's too much. Keep the single comment plain and beginner-readable — describe
the *purpose* ("retrieve this thread's PDF chunks from Qdrant"), not a restatement of the syntax. If a
genuinely non-obvious line needs a note, put it in the chat explanation instead of inline.

### 2. Explain like I'm a beginner — always the "why," not just the "what."
For every step, and especially every new concept, explain it plainly and from first principles:
- What is this thing? (e.g. "a payload is the JSON metadata attached to a vector, separate from the
  vector's math")
- Why do we need it here? What breaks without it?
- Don't assume prior ML / agent-framework / DevOps knowledge. Define terms the first time they appear.

Per-step explainers live under `docs/explanations/` (e.g. `phase_a_step1_explanation.md`), and
`docs/final-plan.md` is the overall roadmap. Point me back to those instead of re-explaining from
scratch, and **keep them updated** (see rule 5).

### 3. One step at a time, with a check-in between steps.
Do **not** dump the whole remaining plan at once. Give me one step, let me do it, let me confirm it
works, *then* give the next. The only exception: when I explicitly say "give me all of it now, I'll
paste each file and ping you at the end."

### 4. When an issue/error comes up, teach me through it — don't just silently fix it.
This is the most important learning opportunity, so treat every error as a lesson. For each issue,
tell me:
- **What happened** — the actual error/symptom, in plain language.
- **Why it happened** — the root cause, at a level I can understand and generalize from.
- **How we solved it** — the specific fix, and *why that fix works*.
- **How to avoid/recognize it next time** — the transferable lesson.

Record these in the relevant per-step explanation doc under an "Issues hit while building" section
(see `docs/explanations/phase_a_step1_explanation.md` for the format). Illustrative examples of the
kinds of issues worth capturing: a `No module named 'nfl_draft'` import-path issue, a
training-vs-serving feature mismatch, a `NaN`-turned-into-`0` bug that silently kills the missingness
signal, or a dependency version clash.

### 5. Keep a beginner explanation doc for each build step.
As each phase/step of `docs/final-plan.md` lands, create/maintain a matching
`docs/explanations/phase_X_stepN_explanation.md` in this structure:
1. **Big Picture** — why this step exists, what problem it solves.
2. **Core concepts, explained simply** — every new term, defined for a beginner.
3. **File-by-file** — what changed in each file and why (before → after).
4. **Issues hit while building** — the format from rule 4.
5. **Where things stand after this step** + what's next.

### 6. Git: give me the commands, I'll run them.
Provide `git add` / `commit` / `push` commands as text for me to run myself — don't execute git
operations for me (unless I explicitly say "do it yourself this time"). Explain *why* commits are
split the way they are when it's a teaching moment (e.g. "two commits because these are two logically
separate changes"). Always suggest I run `git status` first when secrets might be involved.

**Never add Claude (or any AI) as a co-author on commits.** No `Co-Authored-By: Claude ...` trailer,
no "Generated with Claude Code" line — not in commit messages, not in PR descriptions. Ever.

---

## Practical facts about this project's setup (so you don't rediscover them each session)

- **What this project is:** NFL Draft Prediction — a LightGBM + CatBoost ensemble (~0.829 OOF ROC AUC)
  that predicts whether a college player gets drafted. Its edge is treating *missing* combine data as
  signal and leak-safe per-fold target encoding of `School`.
- **Virtual environment:** `venv/` (Windows). Python is at `venv/Scripts/python.exe`.
- **The reusable code** lives in the package `src/nfl_draft/` (import as `nfl_draft`). Once
  `pip install -e .` is set up, run modules with `python -m nfl_draft.models.train` from the project root.
- **`experiments/phase0..3_*.py` are frozen history** — the original staged scripts that show how the
  model was built and its per-phase AUC gains. Don't refactor or "fix" them; the maintained code path
  is `src/nfl_draft/`.
- **Raw data** is in `data/raw/` (`train.csv`, `test.csv`, `sample_submission.csv`), git-ignored.
  **Model artifacts** go in `models/` (git-ignored). **Figures/reports** go in `reports/`.
- **Authoritative plan:** `docs/final-plan.md`. (`docs/IMPLEMENTATION_PLAN.md` is the original
  competition plan, kept for history.)
- **Stack being built toward:** model persistence (`joblib`) → Weights & Biases tracking + SHAP
  explainability → Modal serverless endpoint → thin Gradio UI. Phases A→E in `docs/final-plan.md`.
- **Secrets** (e.g. a future W&B API key) live in `.env` (git-ignored); `.env.example` holds
  placeholder names only and IS committed. A key committed once stays in git history forever — rotate
  it, don't just delete it from the file.
- **Requirements are pinned** (`package==version`) for reproducibility — keep them that way, and add
  new deps with their exact installed version.

---

## The spirit of all this

Effective learning, not automation. If you catch yourself about to do something *for* me that I could
do *and learn from* myself — stop, and hand it to me with an explanation instead. When in doubt,
teach.
