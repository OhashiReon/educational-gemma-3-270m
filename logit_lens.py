import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download
from safetensors import torch as safetensors_torch

from main import Gemma3ForCausalLM, Gemma3TextConfig


@dataclass
class TokenLayerProbs:
    token: str
    token_id: int
    layer_probs: List[float]


@dataclass
class GeneratedToken:
    token: str
    token_id: int


@dataclass
class StepResult:
    step: int
    input_text: str
    important_tokens: List[TokenLayerProbs]
    generated_token: GeneratedToken


@dataclass
class AttentionWeights:
    tokens: List[str]
    weights: List[List[List[List[float]]]]  # [layer][head][query][key]


@dataclass
class LogitLensResult:
    initial_text: str
    steps: List[StepResult]
    final_attention: Optional[AttentionWeights] = None


class LogitLensEngine:
    def __init__(self, model: Gemma3ForCausalLM, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self._captured_hidden: Dict[int, torch.Tensor] = {}
        self._captured_attn: Dict[int, torch.Tensor] = {}

    def _get_hidden_hook(self, layer_idx: int):
        """Hook to capture output hidden states of a layer."""

        def hook(module, input, output):
            self._captured_hidden[layer_idx] = output.detach()

        return hook

    def _get_attn_hook(self, layer_idx: int):
        """Hook to capture attention weights."""

        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                weights = output[1]
                if weights is not None:
                    self._captured_attn[layer_idx] = weights.detach().cpu()

        return hook

    def register_hooks(self):
        """Attach hooks to model layers."""
        for i, layer in enumerate(self.model.model.layers):
            self.hooks.append(layer.register_forward_hook(self._get_hidden_hook(i)))
            self.hooks.append(
                layer.self_attn.register_forward_hook(self._get_attn_hook(i))
            )

    def remove_hooks(self):
        """Clean up hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self._clear_storage()

    def _clear_storage(self):
        self._captured_hidden = {}
        self._captured_attn = {}

    def compute_lens_at_step(self, topk: int = 5) -> List[TokenLayerProbs]:
        """
        Perform Logit Lens analysis using vectorized operations.
        Converts hidden states from all layers to probability distributions.
        """
        num_layers = len(self.model.model.layers)
        if len(self._captured_hidden) != num_layers:
            return []
        hidden_stack = torch.stack(
            [self._captured_hidden[i][:, -1, :] for i in range(num_layers)]
        ).squeeze(1)
        with torch.no_grad():
            normed = self.model.model.norm(hidden_stack)
            logits = self.model.lm_head(normed)
            probs = F.softmax(logits, dim=-1)
        _, topk_indices = torch.topk(probs, topk, dim=-1)
        important_ids = torch.unique(topk_indices).tolist()
        target_probs = probs[:, important_ids].cpu().numpy()
        results = []
        for i, token_id in enumerate(important_ids):
            layer_probs_list = target_probs[:, i].tolist()
            results.append(
                TokenLayerProbs(
                    token=self.tokenizer.decode([token_id]),
                    token_id=token_id,
                    layer_probs=layer_probs_list,
                )
            )
        results.sort(key=lambda x: x.layer_probs[-1], reverse=True)
        return results

    def extract_final_attention(
        self, text: str, max_tokens: int = 30
    ) -> Optional[AttentionWeights]:
        """Format captured attention weights for visualization."""
        if not self._captured_attn:
            return None

        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[-max_tokens:]
        seq_len = len(tokens)

        serialized_weights = []
        num_layers = len(self.model.model.layers)

        for i in range(num_layers):
            if i not in self._captured_attn:
                serialized_weights.append([])
                continue
            attn = self._captured_attn[i]
            current_q_len = attn.shape[2]

            start_idx = max(0, current_q_len - seq_len)

            layer_data = []
            num_heads = attn.shape[1]

            for h in range(num_heads):
                matrix = attn[0, h, start_idx:, start_idx:].tolist()
                layer_data.append(matrix)

            serialized_weights.append(layer_data)

        return AttentionWeights(tokens=tokens, weights=serialized_weights)


def load_gemma_model(repo_id: str):
    """Load model with forced eager attention for visualization."""
    print(f"Loading {repo_id}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)

    # IMPORTANT: Force 'eager' implementation to get attention weights.
    # SDPA (Flash Attention) does not return weights.
    config = Gemma3TextConfig.from_pretrained(repo_id, attn_implementation="eager")

    model = Gemma3ForCausalLM(config)
    weight_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    weight = safetensors_torch.load_file(weight_path)
    model.load_state_dict(weight, strict=False)
    model.tie_weights()
    model.eval()
    return model, tokenizer


def run_logit_lens_pipeline(repo_id: str, prompt: str, steps: int, topk_per_layer: int):
    model, tokenizer = load_gemma_model(repo_id)
    engine = LogitLensEngine(model, tokenizer)
    engine.register_hooks()

    current_text = prompt
    step_results = []
    final_attn = None

    print(f"Starting generation: '{prompt}'")

    with torch.no_grad():
        for step_idx in range(steps):
            inputs = tokenizer(current_text, return_tensors="pt")
            logits = model(**inputs)
            next_logits = logits[:, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1).item()
            next_token = tokenizer.decode([next_id])
            important_tokens = engine.compute_lens_at_step(topk=topk_per_layer)
            step_results.append(
                StepResult(
                    step=step_idx,
                    input_text=current_text,
                    important_tokens=important_tokens,
                    generated_token=GeneratedToken(next_token, int(next_id)),
                )
            )
            if step_idx == steps - 1:
                final_attn = engine.extract_final_attention(current_text)
            engine._clear_storage()
            current_text += next_token
            print(f"Step {step_idx + 1}/{steps}: {next_token!r}")
    engine.remove_hooks()
    return LogitLensResult(
        initial_text=prompt, steps=step_results, final_attention=final_attn
    ), current_text


if __name__ == "__main__":
    from pathlib import Path

    REPO_ID = "google/gemma-3-270m"
    PROMPT = "Hello, my dog is cute"
    STEPS = 10
    OUTPUT_DIR = Path(__file__).parent / "out"
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / "logit_lens.json"

    result, final_text = run_logit_lens_pipeline(
        repo_id=REPO_ID, prompt=PROMPT, steps=STEPS, topk_per_layer=3
    )
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    print(f"\nAnalysis saved to {OUTPUT_FILE}")
    print(f"Final Text: {final_text}")
