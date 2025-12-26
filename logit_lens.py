import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, Set

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
class TokenLayerProbs:
    token: str
    token_id: int
    layer_probs: List[float]


@dataclass
class AttentionWeights:
    tokens: List[str]
    weights: List[List[List[List[float]]]]


@dataclass
class GeneratedToken:
    token: str
    token_id: int | float


@dataclass
class StepResult:
    step: int
    input_text: str
    important_tokens: List[TokenLayerProbs]
    generated_token: GeneratedToken


@dataclass
class LogitLensResult:
    initial_text: str
    steps: List[StepResult]
    final_attention: Optional[AttentionWeights] = None


def get_logits_for_all_layers(
    model, hidden_states: List[torch.Tensor]
) -> List[torch.Tensor]:
    """各レイヤーのhidden stateからlogitsを計算"""
    all_logits = []
    for hidden_state in hidden_states:
        x = hidden_state[:, -1:, :]
        x = model.model.norm(x)
        logits = model.lm_head(x)[:, -1, :]
        all_logits.append(logits)
    return all_logits


def collect_important_token_ids(
    all_logits: List[torch.Tensor], topk_per_layer: int
) -> Set[int]:
    """各レイヤーのTOPKに入るトークンIDを収集"""
    important_ids = set()
    for logits in all_logits:
        probs = F.softmax(logits, dim=-1)
        _, top_ids = torch.topk(probs, topk_per_layer, dim=-1)
        important_ids.update(top_ids[0].tolist())
    return important_ids


def compute_token_layer_probs(
    all_logits: List[torch.Tensor],
    important_token_ids: Set[int],
    tokenizer,
) -> List[TokenLayerProbs]:
    """重要トークンについて全レイヤーの確率を計算"""
    results = []

    for token_id in important_token_ids:
        layer_probs = []
        for logits in all_logits:
            probs = F.softmax(logits, dim=-1)
            prob = probs[0, token_id].item()
            layer_probs.append(prob)

        results.append(
            TokenLayerProbs(
                token=tokenizer.decode([token_id]),
                token_id=token_id,
                layer_probs=layer_probs,
            )
        )

    results.sort(key=lambda x: x.layer_probs[-1], reverse=True)
    return results


def extract_attention_weights(
    all_self_attn_weights: Tuple[torch.Tensor],
    tokenizer,
    text: str,
    max_tokens: int = 30,
    batch_idx: int = 0,
) -> AttentionWeights:
    """attention weightsをシリアライズ可能な形式に変換"""
    tokens = tokenizer.tokenize(text)[:max_tokens]
    T = len(tokens)

    weights = []
    for layer_idx in range(len(all_self_attn_weights)):
        attn = all_self_attn_weights[layer_idx]
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu()

        layer_weights = []
        num_heads = attn.shape[1]
        for head_idx in range(num_heads):
            mat = attn[batch_idx, head_idx, :T, :T].tolist()
            layer_weights.append(mat)

        weights.append(layer_weights)

    return AttentionWeights(tokens=tokens, weights=weights)


def run_logit_lens(
    *,
    model,
    tokenizer,
    text: str,
    N: int,
    topk_per_layer: int = 3,
    max_attention_tokens: int = 30,
) -> Tuple[LogitLensResult, str]:
    """
    Args:
        model: モデル
        tokenizer: トークナイザー
        text: 入力テキスト
        N: 生成ステップ数
        topk_per_layer: 各レイヤーで保存する上位K個のトークン数
        max_attention_tokens: attention可視化用の最大トークン数
    """
    steps: List[StepResult] = []
    current_text = text
    final_attention = None

    for step_idx in range(N):
        inputs = tokenizer(current_text, return_tensors="pt")
        final_logits, all_hidden_states, all_self_attn_weights = model(**inputs)

        all_logits = get_logits_for_all_layers(model, all_hidden_states)

        important_token_ids = collect_important_token_ids(all_logits, topk_per_layer)

        important_tokens = compute_token_layer_probs(
            all_logits, important_token_ids, tokenizer
        )

        if step_idx == N - 1:
            final_attention = extract_attention_weights(
                all_self_attn_weights,
                tokenizer,
                current_text,
                max_tokens=max_attention_tokens,
            )

        final_probs = F.softmax(final_logits[:, -1, :], dim=-1)
        next_id = torch.argmax(final_probs, dim=-1).item()
        next_token = tokenizer.decode([next_id])

        steps.append(
            StepResult(
                step=step_idx,
                input_text=current_text,
                important_tokens=important_tokens,
                generated_token=GeneratedToken(
                    token=next_token,
                    token_id=next_id,
                ),
            )
        )

        current_text += next_token

    return (
        LogitLensResult(
            initial_text=text, steps=steps, final_attention=final_attention
        ),
        current_text,
    )


if __name__ == "__main__":
    model, tokenizer = load_model("google/gemma-3-270m")
    result, final_text = run_logit_lens(
        model=model,
        tokenizer=tokenizer,
        text="Hello, my dog is cute",
        N=10,
        topk_per_layer=3,
        max_attention_tokens=30,
    )

    with open("logit_lens.json", "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    print(f"Generated text: {final_text}")
