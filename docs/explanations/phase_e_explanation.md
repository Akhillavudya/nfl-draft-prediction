# Phase E — Gradio UI (a thin client to the Modal endpoint)

## 1. Big Picture
Phase D put the model behind a public URL, but talking to it still means hand-writing JSON and running
`curl`. **Phase E adds the human face**: a small web form where anyone types a player's Combine stats
and sees a draft probability plus *why*. Crucially, the form does **no machine learning of its own** —
it just collects fields, POSTs them to the Modal endpoint, and draws the answer. The intelligence stays
in the serverless API; the UI is disposable. That separation (a *model service* + a *thin frontend*) is
exactly how real ML products are built, and it keeps Modal — the more impressive, portfolio-worthy
piece — as the showpiece rather than burying it inside a monolithic app.

## 2. Core concepts, explained simply
- **Gradio** — a Python library that turns a function into a web UI. You describe the inputs (a number
  box, a dropdown) and outputs (a label, a plot); Gradio builds the page and runs a small web server.
  No HTML/JS required.
- **Thin client** — a frontend that holds no business logic. Ours never loads a model or imports
  `predict_one`; it only formats a request and renders a response. "Thin" because all the weight lives
  elsewhere (on Modal). The opposite would be a "fat" app that loads the models itself — then you'd be
  hosting the ML twice and Modal would be pointless.
- **`gr.Blocks`** — Gradio's flexible layout container. Inside it you place components in rows/columns
  and then *wire* a button to a function: `submit.click(fn, inputs=[...], outputs=[...])` means "when
  clicked, call `fn` with the current values of these inputs and send its return values to these
  outputs."
- **Component ↔ argument order** — the list you pass to `inputs=` maps *positionally* to your
  function's parameters, and the function's return tuple maps positionally to `outputs=`. Order is the
  contract; get it wrong and the wrong box feeds the wrong argument.
- **`gr.Number` and blank = `None`** — an empty number box returns Python `None`, not `0`. That's the
  hook we use to preserve the model's missing-data signal: we drop `None` values from the payload so the
  field arrives *absent* → server-side `NaN` → the `_missing` flags fire. If Gradio had defaulted blanks
  to `0`, the UI would silently destroy the model's whole edge.
- **`allow_custom_value=True`** on a dropdown — lets the user submit a value that isn't in the list
  (e.g. a school not in `train.csv`), instead of forcing a choice. Graceful degradation.
- **Environment variable for the URL** — the endpoint is read from `MODAL_ENDPOINT_URL`, falling back to
  the deployed default. Config that varies by environment (dev vs. your own deploy) belongs in an env
  var, not hard-coded.

## 3. File-by-file
- `app/gradio_app.py` — **new.** The whole UI. Key parts:
  - `MODAL_URL` — endpoint from env var, else the deployed default.
  - `_choices()` — reads `data/raw/train.csv` once to fill the School / Player_Type / Position_Type /
    Position dropdowns from real training values (236 schools, etc.); returns empty lists if the CSV
    isn't present, so the app still runs anywhere.
  - `predict(...)` — packs the 9 numeric + 4 categorical inputs into the endpoint's JSON, **omitting
    blank numerics** so they stay `NaN`, POSTs with `requests`, and returns `(label_dict, figure)`.
    A network/HTTP failure is surfaced to the user via `gr.Error`.
  - `_factor_plot(...)` — a small horizontal SHAP bar (green raises draft odds, red lowers) built from
    the endpoint's `top_factors`.
  - `build_ui()` — lays out the form (inputs + Predict button on the left, probability label + plot on
    the right) and wires the click. `launch()` serves it locally.
- `pyproject.toml`, `requirements.txt` — **edited.** Pinned `gradio==6.19.0` and `requests==2.34.2`
  (the exact installed versions).
- `.gitignore` — **edited.** Ignore Gradio's runtime `flagged/` and `.gradio/` directories.
- `README.md` — **edited.** Added a "Web UI (Gradio)" section and ticked the Phase E roadmap box.

## 4. Issues hit while building
- **Why blanks had to be *omitted*, not sent as `null`/`0`.**
  - *What could have happened:* the natural first instinct is to send every form field, defaulting empty
    Combine boxes to `0`. That player would score high, because `0` reads as a real (poor-but-present)
    measurement — the exact training-vs-serving bug this whole project guards against.
  - *Why:* the model's edge is that *missingness itself* is signal (`Age_missing`, `num_drills_missing`).
    A present `0` and an absent `NaN` are opposite inputs.
  - *How we handled it:* `gr.Number` yields `None` for a blank box, and `predict()` drops `None`
    numerics from the payload, so they arrive absent → `NaN`. Verified end-to-end: full combine → 0.856,
    same player with all drills blank → 0.071 (matching the README's ~0.85 / ~0.07).
  - *Recognise next time:* whenever a UI feeds a model, ask "what does an empty field become?" — the safe
    default for a signal-carrying missing value is *absent*, never a stand-in number.

## 5. Where things stand + what's next
The full stack is live: **Gradio form → JSON (blanks dropped) → HTTP → Modal → LightGBM+CatBoost blend →
SHAP → back to the form.** Phases A–E of `docs/final-plan.md` are all done: structured repo, installable
package, persisted model, W&B tracking + SHAP, Modal endpoint, and now a UI. Optional next steps (not in
the plan): host the Gradio app on Hugging Face Spaces as a public thin client, and fill in the
leaderboard placeholder once the rank lands (~mid-July 2026).
