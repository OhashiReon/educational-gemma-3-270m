import json
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("Logit Lens Explorer")

# ============================
# Load JSON
# ============================
json_file = st.file_uploader("Upload logit_lens.json", type=["json"])
if json_file is None:
    st.stop()

data: dict[str, Any] = json.load(json_file)
steps = data["steps"]

# ============================
# Step selector
# ============================
step_idx = st.slider("Step", 0, len(steps) - 1, value=0)
step = steps[step_idx]

st.markdown("### Context")
st.code(step["input_text"], language="text")
st.markdown(f"**Generated token:** `{step['generated_token']['token']}`")

layers = step["layers"]

# ============================
# Flatten layer/topk into DataFrame
# ============================
rows = []
for l in layers:
    for tp in l["topk"]:
        rows.append(
            {
                "layer": l["layer"],
                "token": tp["token"],
                "prob": tp["prob"],
            }
        )

df = pd.DataFrame(rows)

# ============================
# Section 1: Heatmap (ALL layers at once)
# ============================
st.markdown("## Logit Lens Heatmap (Layers × Tokens)")

# Token selection rule:
# 1. For each token, compute max(prob) across all layers
# 2. Take top-N tokens by that value
TOP_K_PER_LAYER = 3

per_layer_top = (
    df.sort_values(["layer", "prob"], ascending=[True, False])
    .groupby("layer")
    .head(TOP_K_PER_LAYER)
)

top_tokens = per_layer_top["token"].unique()

heatmap_df = df[df["token"].isin(top_tokens)]

heatmap = (
    alt.Chart(heatmap_df)
    .mark_rect()
    .encode(
        x=alt.X("layer:O", title="Layer"),
        y=alt.Y("token:N", sort="-x", title="Token"),
        color=alt.Color("prob:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["layer", "token", "prob"],
    )
    .properties(height=400)
)

st.altair_chart(heatmap, width="stretch")

st.caption(
    "Heatmap tokens are selected by: for each layer, take top 3 tokens by probability; then combine these tokens across all layers."
)


# ============================
# Section 2: Probability evolution (automatic token choice)
# ============================
st.markdown("## Probability Evolution Across Layers")

# Token selection rule:
# Take top-N tokens from the FINAL layer's probabilities
TOP_N_LINES = 5

final_layer = df["layer"].max()
final_df = df[df["layer"] == final_layer]

line_tokens = (
    final_df.sort_values("prob", ascending=False).head(TOP_N_LINES)["token"].tolist()
)

line_df = df[df["token"].isin(line_tokens)]

line_chart = (
    alt.Chart(line_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("layer:Q", title="Layer"),
        y=alt.Y("prob:Q", title="Probability"),
        color=alt.Color("token:N", legend=alt.Legend(title="Token")),
        tooltip=["layer", "token", "prob"],
    )
    .properties(height=300)
)

st.altair_chart(line_chart, use_container_width=True)

st.caption(
    "Line chart tokens are selected from the FINAL layer (top 5 by probability)."
)

# ============================
# Section 3: Raw per-layer logit lens table (full detail)
# ============================
st.markdown("## Per-layer Logit Lens (Raw Data)")

st.caption(
    "This table shows all recorded (layer, token, probability) rows for the selected step."
)

st.dataframe(
    df.sort_values(["layer", "prob"], ascending=[True, False]),
    use_container_width=True,
)
