import copy
import functools
import math
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3PreTrainedModel,
    Gemma3TextConfig,
)


class GELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://huggingface.co/papers/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self, use_gelu_tanh_python: bool = False):
        super().__init__()
        if use_gelu_tanh_python:
            self.act = self._gelu_tanh_python
        else:
            self.act = functools.partial(nn.functional.gelu, approximate="tanh")

    def _gelu_tanh_python(self, input: torch.Tensor) -> torch.Tensor:
        return (
            input
            * 0.5
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act(input)


ACT2FN = {
    "gelu_pytorch_tanh": GELUTanh,
}


def create_causal_mask(
    config: Gemma3TextConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
) -> torch.Tensor:
    batch_size, query_length, _ = input_embeds.shape
    device = input_embeds.device
    dtype = input_embeds.dtype
    past_length = 0
    key_length = past_length + query_length
    query_pos = cache_position.view(query_length, 1)
    key_pos = torch.arange(key_length, device=device).view(1, key_length)
    mask = torch.zeros((query_length, key_length), device=device, dtype=dtype)
    mask.masked_fill_(key_pos > query_pos, torch.finfo(dtype).min)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        padding_mask = torch.zeros_like(attention_mask, dtype=dtype)
        padding_mask.masked_fill_(attention_mask == 0, torch.finfo(dtype).min)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        mask = mask + padding_mask
    return mask


def create_sliding_window_causal_mask(
    config: Gemma3TextConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
) -> torch.Tensor:
    causal_mask = create_causal_mask(
        config, input_embeds, attention_mask, cache_position
    )
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError(
            "Config must have `sliding_window` for sliding window attention."
        )
    batch_size, query_length, _ = input_embeds.shape
    device = input_embeds.device
    dtype = input_embeds.dtype
    past_length = 0
    key_length = past_length + query_length
    query_pos = cache_position.view(query_length, 1)
    key_pos = torch.arange(key_length, device=device).view(1, key_length)
    dist = query_pos - key_pos
    window_mask = torch.zeros((query_length, key_length), device=device, dtype=dtype)
    window_mask.masked_fill_(dist > sliding_window, torch.finfo(dtype).min)
    window_mask = window_mask.unsqueeze(0).unsqueeze(0)
    combined_mask = causal_mask + window_mask
    return combined_mask


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [batch, seq_len, head_dim] or [1, seq_len, head_dim]
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma3TextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Instantiate the activation module; mapping stores a class, not a callable.
        self.act_fn = ACT2FN[config.hidden_activation]()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = not self.config.use_bidirectional_attention

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs


class Gemma3TextModel(Gemma3PreTrainedModel):
    config: Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        # 1. Embeddings
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )

        # 2. Layers
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # 3. Norm
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 4. RoPE
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)

        cache_position = torch.arange(
            0,
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
        position_ids = cache_position.unsqueeze(0)

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        }

        full_attn_mask = create_causal_mask(**mask_kwargs)
        sliding_attn_mask = create_sliding_window_causal_mask(**mask_kwargs)

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # decoder layers
        for decoder_layer in self.layers:
            if decoder_layer.attention_type == "sliding_attention":
                current_mask = sliding_attn_mask
            else:
                current_mask = full_attn_mask

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=current_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma3ForCausalLM(Gemma3PreTrainedModel):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: torch.Tensor = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(outputs)
        return logits


if __name__ == "__main__":
    import safetensors
    import transformers
    from huggingface_hub import hf_hub_download

    tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-3-270m")
    config = Gemma3TextConfig.from_pretrained("google/gemma-3-270m")
    print(config)
    model = Gemma3ForCausalLM(config)
    weight_path = hf_hub_download(
        repo_id="google/gemma-3-270m", filename="model.safetensors"
    )
    weight = safetensors.torch.load_file(weight_path)
    model.load_state_dict(weight, strict=False)
    model.tie_weights()
    text = "Hello, my dog is cute"
    for _ in range(20):
        logits = model.forward(**tokenizer(text, return_tensors="pt"))
        ids = logits.argmax(-1)
        text = text + tokenizer.decode(ids[0, -1].unsqueeze(0))
        print(text)
