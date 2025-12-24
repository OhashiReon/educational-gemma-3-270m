import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download
from safetensors import torch as safetensors_torch

from main import Gemma3ForCausalLM, Gemma3TextConfig


def load_model(repo_id: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)
    config = Gemma3TextConfig.from_pretrained(repo_id)
    model = Gemma3ForCausalLM(config)
    weight_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    weight = safetensors_torch.load_file(weight_path)
    model.load_state_dict(weight, strict=False)
    model.tie_weights()
    return model, tokenizer


@dataclass
class TokenProb:
    token: str
    token_id: int
    prob: float


@dataclass
class LayerLogitLens:
    layer: int
    topk: List[TokenProb]


@dataclass
class FinalLogits:
    topk: List[TokenProb]


@dataclass
class GeneratedToken:
    token: str
    token_id: int


@dataclass
class StepResult:
    step: int
    input_text: str
    layers: List[LayerLogitLens]
    final: FinalLogits
    generated_token: GeneratedToken


@dataclass
class LogitLensResult:
    initial_text: str
    steps: List[StepResult]


def topk_from_logits(logits: torch.Tensor, tokenizer, k: int) -> List[TokenProb]:
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k, dim=-1)
    return [
        TokenProb(
            token=tokenizer.decode([tid.item()]),
            token_id=tid.item(),
            prob=float(p.item()),
        )
        for p, tid in zip(top_probs[0], top_ids[0])
    ]


def run_logit_lens(
    *,
    model,
    tokenizer,
    text: str,
    N: int,
    topk: int,
) -> Tuple[LogitLensResult, Tuple[Optional[torch.Tensor]], str]:
    steps: List[StepResult] = []
    current_text = text

    for step_idx in range(N):
        inputs = tokenizer(current_text, return_tensors="pt")
        final_logits, all_hidden_states, all_self_attn_weights = model(**inputs)

        layers: List[LayerLogitLens] = []
        for layer_idx, hidden_state in enumerate(all_hidden_states):
            x = hidden_state[:, -1:, :]
            x = model.model.norm(x)
            logits = model.lm_head(x)[:, -1, :]
            layers.append(
                LayerLogitLens(
                    layer=layer_idx,
                    topk=topk_from_logits(logits, tokenizer, topk),
                )
            )

        final = FinalLogits(
            topk=topk_from_logits(
                final_logits[:, -1, :],
                tokenizer,
                topk,
            )
        )

        next_id = final.topk[0].token_id
        next_token = final.topk[0].token

        steps.append(
            StepResult(
                step=step_idx,
                input_text=current_text,
                layers=layers,
                final=final,
                generated_token=GeneratedToken(
                    token=next_token,
                    token_id=next_id,
                ),
            )
        )

        current_text += next_token

    return (
        LogitLensResult(initial_text=text, steps=steps),
        all_self_attn_weights,
        current_text,
    )


def plot_attention_all_layers_heads_qk(
    *,
    all_self_attn_weights,
    tokenizer,
    text: str,
    max_tokens: int = 30,
    batch_idx: int = 0,
):
    tokens = tokenizer.tokenize(text)[:max_tokens]
    T = len(tokens)

    num_layers = len(all_self_attn_weights)
    num_heads = all_self_attn_weights[0].shape[1]

    fig, axes = plt.subplots(
        num_layers,
        num_heads,
        figsize=(2.2 * num_heads, 2.2 * num_layers),
        squeeze=False,
    )

    for layer_idx in range(num_layers):
        attn = all_self_attn_weights[layer_idx]
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu()

        for head_idx in range(num_heads):
            ax = axes[layer_idx][head_idx]
            mat = attn[batch_idx, head_idx, :T, :T]
            ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticklabels(tokens, fontsize=6)
            if layer_idx == 0:
                ax.set_title(f"H{head_idx}", fontsize=9)
            if head_idx == 0:
                ax.set_ylabel(f"L{layer_idx}", fontsize=9)

    plt.tight_layout()
    plt.savefig("attention.png", dpi=300)


if __name__ == "__main__":
    model, tokenizer = load_model("google/gemma-3-270m")
    result, all_self_attn_weights, final_text = run_logit_lens(
        model=model,
        tokenizer=tokenizer,
        text="Hello, my dog is cute",
        N=10,
        topk=100,
    )

    with open("logit_lens.json", "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    plot_attention_all_layers_heads_qk(
        all_self_attn_weights=all_self_attn_weights,
        tokenizer=tokenizer,
        text=final_text,
    )
