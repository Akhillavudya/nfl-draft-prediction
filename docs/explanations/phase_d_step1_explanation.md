# Phase D · Step 1 — Add, install & authenticate Modal

## 1. Big Picture
Phases B and C gave us a *deployable* model (saved artifacts + a `predict_one` function) and made it
*explainable*. But it still only runs on your laptop. **Phase D deploys it** — puts it behind a public
URL so anyone can send a player's stats and get a probability back, with no machine of yours involved.
Step 1 is the groundwork: pick the platform (**Modal**), add it as a pinned dependency, install it, and
log in. No serving code yet — just getting the tool in the box and authenticated.

## 2. Core concepts, explained simply
- **Serverless** — you don't rent or manage a server. You hand Modal a Python function + a recipe for
  its environment; Modal spins up a container **only when a request arrives**, runs it, then spins it
  down. Idle = nothing running = (basically) free. The opposite of a VM you pay for 24/7.
- **Modal** — a serverless platform aimed at Python/ML. New and modern — deliberately chosen so this
  project shows a platform the rest of the portfolio doesn't (avoiding the Streamlit/HF monoculture).
- **The three Modal building blocks** (you'll meet them in Step 2):
  - **Image** — the recipe for the container: OS + installed packages + files copied in.
  - **App / Function** — your code, labelled so Modal can run it in the cloud.
  - **Web endpoint** — a public URL that triggers a Function over HTTP.
- **Authentication token** — Modal needs to know *whose* account is deploying. `modal setup` opens a
  browser, logs you in, and writes a token to `~/.modal.toml`. That token (not a password) is what the
  CLI uses from then on.
- **Pinning a new dependency** — we record the *exact* installed version (`modal==1.5.1`) in
  `pyproject.toml` and `requirements.txt`, so anyone rebuilding the project gets the same Modal.

## 3. File-by-file
- `pyproject.toml` — **edited.** Added `"modal==1.5.1"` to `dependencies`.
- `requirements.txt` — **edited.** Added `modal==1.5.1` under a "Serverless deployment (Phase D)" note.
- *(no code files yet — installation + login are environment setup, not source changes.)*

Commands run:
- `pip install modal` → resolved **modal 1.5.1** (then pinned it, per the project's "pin the exact
  installed version" rule).
- `modal setup` → browser login, token written to `C:\Users\91888/.modal.toml`, workspace
  `akhillavudya4567` verified.

## 4. Issues hit while building
- **`pip install modal` downgraded `protobuf` 7.35.1 → 6.33.6.**
  - *What happened:* Modal depends on an older protobuf range, so pip replaced the newer one already in
    the venv.
  - *Why it matters:* another package (W&B, Phase C) may also care about protobuf — this is a classic
    **dependency clash**, where two libraries want different versions of a shared sub-dependency.
  - *How we handled it:* noted it and moved on — protobuf 6.x is fine for serving, and Phase D doesn't
    touch W&B. The lesson: after installing a big new dep, *read pip's output* for "Uninstalling …"
    lines; a surprise downgrade can break an unrelated feature later.
  - *Recognise next time:* if W&B ever errors with a protobuf message, this is the cause — reinstall a
    compatible protobuf or run the two tools in separate environments.

## 5. Where things stand + what's next
Modal is installed, pinned, and authenticated to the `akhillavudya4567` workspace. **Next (Step 2):**
write `app/modal_app.py` — the container image recipe plus a `POST /predict` endpoint that wraps
`predict_one` — and test it live with `modal serve`.
