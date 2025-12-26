import json
from typing import Any
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("Logit Lens Explorer")

DEFAULT_JSON_PATH = Path(__file__).parent / "out" / "logit_lens.json"
print(DEFAULT_JSON_PATH)
if not DEFAULT_JSON_PATH.exists():
    json_file = st.file_uploader("Upload logit_lens.json", type=["json"])
    if json_file is None:
        st.warning("Please upload a JSON file or run 'logit_lens.py' to generate one.")
        st.stop()
    else:
        data = json.load(json_file)
else:
    json_file_uploaded = st.file_uploader("Upload logit_lens.json", type=["json"])
    with open(DEFAULT_JSON_PATH, "r", encoding="utf-8") as f:
        json_file_default = f
        data = json.load(json_file_default)
    if json_file_uploaded is None:
        st.info(
            "Using default logit_lens.json from 'out' directory. if use upload your own file, please re-upload from the uploader above."
        )
        json_file = json_file_default
    else:
        st.info("Using uploaded logit_lens.json file.")
        json_file = json_file_uploaded
        data = json.load(json_file)

steps = data["steps"]


step_idx = st.slider("Step", 0, len(steps) - 1, value=0)
step = steps[step_idx]
st.caption(
    "Select a step for each generated token. "
    "Moving the slider will show the internal state of the model at that point."
)
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


st.markdown("## Logit Lens Heatmap")
st.caption(
    "The vertical axis represents the dominant token, the horizontal axis represents the layer, and the color represents the probability. "
    "The tokens are sorted by the layer in which they first became dominant (highest ranked), so you can see at what stage the model started to predict what. "
)
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


st.markdown("## Logit Lens Line Chart")
st.caption(
    "This visualizes how the probability of each token changes as the layers progress, and shows how the probability increases suddenly at certain layers."
)
line_tokens = sorted_tokens[:]

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

with st.expander("ℹ What is Logit Lens?"):
    st.markdown("""
    **Logit Lens** is a technique to visualize the internal "thought process" of a Transformer model.
    
    Normally, a model only converts its hidden state to a vocabulary probability distribution at the very last layer. 
    Logit Lens applies this final conversion (using the explicit `lm_head` and `RMSNorm`) to the **intermediate hidden states** of every layer.
    
    - **Vertical Axis**: Tokens that ranked in the top probabilities.
    - **Horizontal Axis**: The layer number (0 to Final).
    - **Color**: The probability assigned to that token at that layer.
    
    This allows us to see *when* the model becomes confident about a specific token.
                
    see the [this article](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) for more details.

    """)


if data.get("final_attention") is not None:
    st.markdown("## Attention Weights Visualization")

    attention = data["final_attention"]
    tokens = attention["tokens"]
    weights = attention["weights"]

    num_layers = len(weights)
    num_heads = len(weights[0]) if num_layers > 0 else 0

    st.caption(
        f"Showing attention weights for all {num_layers} layers and {num_heads} heads"
    )

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

with st.expander("ℹ️ What are Attention Weights?"):
    st.markdown("""
    **Self-Attention** is the mechanism allows the model to relate different positions of the input sequence to compute a representation of the sequence.
    
    - **Query (Y-axis)**: The token currently being processed.
    - **Key (X-axis)**: The context tokens being attended to.
    - **Color Intensity**: Represents the attention weight (importance). A higher weight means the model is "focusing" more on that Key token to produce the Query token's representation.
    
    Gemma 3 uses Multi-Head Attention, allowing it to focus on different aspects of relationships simultaneously (visualized here as separate grids per head).
    """)
