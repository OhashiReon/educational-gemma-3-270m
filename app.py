import json
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("Logit Lens Explorer")


json_file = st.file_uploader("Upload logit_lens.json", type=["json"])
if json_file is None:
    st.stop()

data: dict[str, Any] = json.load(json_file)
steps = data["steps"]


step_idx = st.slider("Step", 0, len(steps) - 1, value=0)
step = steps[step_idx]

st.markdown("### Context")
st.code(step["input_text"], language="text")
st.markdown(f"**Generated token:** `{step['generated_token']['token']}`")


rows = []
for token_info in step["important_tokens"]:
    token = token_info["token"]
    token_id = token_info["token_id"]
    layer_probs = token_info["layer_probs"]

    for layer_idx, prob in enumerate(layer_probs):
        rows.append(
            {
                "layer": layer_idx,
                "token": token,
                "token_id": token_id,
                "prob": prob,
            }
        )

df = pd.DataFrame(rows)


token_first_layer = {}
token_first_layer_prob = {}

for token in df["token"].unique():
    token_df = df[df["token"] == token].sort_values("layer")

    for _, row in token_df.iterrows():
        layer = row["layer"]
        prob = row["prob"]

        layer_df = df[df["layer"] == layer].sort_values("prob", ascending=False)
        rank = (layer_df["token"] == token).idxmax()
        rank_position = layer_df.index.get_loc(rank)

        if rank_position < 3 and token not in token_first_layer:
            token_first_layer[token] = layer
            token_first_layer_prob[token] = prob
            break

    if token not in token_first_layer:
        max_prob_row = token_df.loc[token_df["prob"].idxmax()]
        token_first_layer[token] = max_prob_row["layer"]
        token_first_layer_prob[token] = max_prob_row["prob"]

df["first_important_layer"] = df["token"].map(token_first_layer)
df["first_layer_prob"] = df["token"].map(token_first_layer_prob)


st.markdown("## Logit Lens Heatmap (Layers × Tokens)")

token_sort_df = (
    df[["token", "first_important_layer", "first_layer_prob"]]
    .drop_duplicates()
    .sort_values(["first_important_layer", "first_layer_prob"], ascending=[True, False])
)

sorted_tokens = token_sort_df["token"].tolist()

heatmap_df = df.copy()
heatmap_df["token"] = pd.Categorical(
    heatmap_df["token"], categories=sorted_tokens, ordered=True
)

heatmap = (
    alt.Chart(heatmap_df)
    .mark_rect()
    .encode(
        x=alt.X("layer:O", title="Layer"),
        y=alt.Y("token:N", sort=sorted_tokens, title="Token"),
        color=alt.Color("prob:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["layer", "token", "prob", "first_important_layer"],
    )
    .properties(height=600)
)

st.altair_chart(heatmap, use_container_width=True)

st.caption(
    "Tokens are sorted by: (1) the layer where they first appeared in TOP3, "
    "(2) their probability in that layer (descending)."
)


st.markdown("## Probability Evolution Across Layers")

TOP_N_LINES = 15
line_tokens = sorted_tokens[:TOP_N_LINES]

line_df = df[df["token"].isin(line_tokens)].copy()

line_chart = (
    alt.Chart(line_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("layer:Q", title="Layer"),
        y=alt.Y("prob:Q", title="Probability"),
        color=alt.Color("token:N", sort=line_tokens, legend=alt.Legend(title="Token")),
        tooltip=["layer", "token", "prob"],
    )
    .properties(height=300)
)

st.altair_chart(line_chart, use_container_width=True)

st.caption(f"Line chart shows top {TOP_N_LINES} tokens (same sort order as heatmap).")


if data.get("final_attention") is not None:
    st.markdown("## Attention Weights Visualization (All Layers × Heads)")

    attention = data["final_attention"]
    tokens = attention["tokens"]
    weights = attention["weights"]

    num_layers = len(weights)
    num_heads = len(weights[0]) if num_layers > 0 else 0

    st.caption(
        f"Showing attention weights for all {num_layers} layers and {num_heads} heads"
    )

    with st.expander("Show token list"):
        token_df = pd.DataFrame({"index": range(len(tokens)), "token": tokens})
        st.dataframe(token_df, use_container_width=True)

    for layer_idx in range(num_layers):
        st.markdown(f"### Layer {layer_idx}")

        layer_rows = []
        for head_idx in range(num_heads):
            attn_matrix = weights[layer_idx][head_idx]
            for q_idx, q_token in enumerate(tokens):
                for k_idx, k_token in enumerate(tokens):
                    layer_rows.append(
                        {
                            "head": f"H{head_idx}",
                            "query_idx": q_idx,
                            "query_token": q_token,
                            "key_idx": k_idx,
                            "key_token": k_token,
                            "weight": attn_matrix[q_idx][k_idx],
                        }
                    )

        layer_df = pd.DataFrame(layer_rows)

        base = (
            alt.Chart(layer_df)
            .mark_rect()
            .encode(
                x=alt.X("key_idx:O", axis=None),
                y=alt.Y("query_idx:O", axis=None),
                color=alt.Color(
                    "weight:Q", scale=alt.Scale(scheme="viridis"), legend=None
                ),
                tooltip=[
                    "head",
                    "query_token",
                    "key_token",
                    alt.Tooltip("weight", format=".3f"),
                ],
            )
            .properties(
                width=240,
                height=240,
            )
        )

        chart = base.facet(
            column=alt.Column("head:N", title=None, header=alt.Header(title=None)),
            columns=6,
        ).configure_view(
            stroke=None,
        )

        st.altair_chart(chart, use_container_width=True)


st.markdown("## Per-layer Logit Lens (Raw Data)")

st.caption(
    "This table shows all important tokens tracked across all layers for the selected step."
)

display_df = df.copy()
display_df["token"] = pd.Categorical(
    display_df["token"], categories=sorted_tokens, ordered=True
)
display_df = display_df.sort_values(["token", "layer"])

st.dataframe(
    display_df[["token", "layer", "prob", "first_important_layer"]],
    use_container_width=True,
)


st.markdown("## Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Number of layers", df["layer"].nunique())

with col2:
    st.metric("Number of tracked tokens", df["token"].nunique())

with col3:
    st.metric("Total data points", len(df))
