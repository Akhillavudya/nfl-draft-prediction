"""Gradio UI: a thin client that POSTs a player's raw stats to the Modal endpoint."""
import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")  # render figures to memory, no GUI window
import matplotlib.pyplot as plt
import pandas as pd
import requests

# Where the live model lives; override with MODAL_ENDPOINT_URL to point at your own deploy.
MODAL_URL = os.environ.get(
    "MODAL_ENDPOINT_URL",
    "https://akhillavudya4567--nfl-draft-prediction-predict.modal.run",
)

# Raw numeric combine fields — a blank one means "missing", so it must never become 0.
NUMERIC_FIELDS = ["Age", "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
                  "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle"]
# Raw categorical fields — offered as dropdowns filled from the training set.
CATEGORICAL_FIELDS = ["School", "Player_Type", "Position_Type", "Position"]

# The training CSV lives two levels up from this file (repo_root/data/raw/train.csv).
TRAIN_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.csv"

# A full-combine example matching the README curl (~0.85) as sensible form defaults.
DEFAULTS = {"Age": 22, "Height": 74, "Weight": 210, "Sprint_40yd": 4.5,
            "Vertical_Jump": 35, "Bench_Press_Reps": 18, "Broad_Jump": 120,
            "Agility_3cone": 6.9, "Shuttle": 4.2, "School": "Alabama",
            "Player_Type": "offense", "Position_Type": "backs_receivers", "Position": "WR"}


def _choices():
    """Read each categorical column's distinct training values for the dropdowns (empty if no CSV)."""
    try:
        df = pd.read_csv(TRAIN_CSV)
        return {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL_FIELDS}
    except Exception:
        return {c: [] for c in CATEGORICAL_FIELDS}


CHOICES = _choices()


def _factor_plot(top_factors):
    """Draw a small horizontal bar of the top SHAP factors (green raises, red lowers the odds)."""
    labels = [f["feature"] for f in top_factors][::-1]
    shaps = [f["shap"] for f in top_factors][::-1]
    colors = ["#2e7d32" if s > 0 else "#c62828" for s in shaps]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.barh(labels, shaps, color=colors)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_title("Top factors behind this prediction (SHAP)")
    ax.set_xlabel("← lowers draft odds        raises draft odds →")
    fig.tight_layout()
    return fig


def predict(age, height, weight, sprint, vertical, bench, broad, cone, shuttle,
            school, player_type, position_type, position):
    """Assemble the raw player payload, POST it to Modal, and render probability + SHAP factors."""
    values = [age, height, weight, sprint, vertical, bench, broad, cone, shuttle]
    payload = {f: v for f, v in zip(NUMERIC_FIELDS, values) if v is not None}  # blanks stay absent → NaN
    payload.update({"School": school, "Player_Type": player_type,
                    "Position_Type": position_type, "Position": position})
    try:
        resp = requests.post(MODAL_URL, json=payload, timeout=60)
        resp.raise_for_status()
        out = resp.json()
    except Exception as exc:
        raise gr.Error(f"Could not reach the Modal endpoint: {exc}")
    prob = out["probability"]
    return {"Drafted": prob, "Not drafted": 1 - prob}, _factor_plot(out["top_factors"])


def _num(field):
    """Build a numeric input pre-filled with the example value (clear it to send 'missing')."""
    return gr.Number(label=field, value=DEFAULTS[field])


def _cat(field):
    """Build a dropdown for a categorical field, allowing values outside the training list."""
    return gr.Dropdown(label=field, choices=CHOICES[field], value=DEFAULTS[field],
                       allow_custom_value=True)


def build_ui():
    """Lay out the form, wire the submit button to predict, and return the Gradio app."""
    with gr.Blocks(title="NFL Draft Prediction") as demo:
        gr.Markdown(
            "# NFL Draft Prediction\n"
            "Enter a college player's Combine stats to estimate their draft probability. "
            "**Clearing a Combine field** (leaving it blank) is meaningful — the model treats "
            "missing measurements as a signal, so the probability drops sharply."
        )
        with gr.Row():
            with gr.Column():
                inputs = [_num(f) for f in NUMERIC_FIELDS] + [_cat(f) for f in CATEGORICAL_FIELDS]
                submit = gr.Button("Predict", variant="primary")
            with gr.Column():
                label = gr.Label(label="Draft probability")
                plot = gr.Plot(label="Why")
        submit.click(predict, inputs=inputs, outputs=[label, plot])
    return demo


if __name__ == "__main__":
    build_ui().launch()
