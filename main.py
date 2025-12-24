import copy
import functools
import json
from dataclasses import dataclass
from typing import List, Optional, cast, Callable, Tuple

from jaxtyping import Float, Int

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


@dataclass
class Gemma3TextConfig:
    """Configuration for the Gemma3 text model.

    Contains vocabulary and model dimension settings, attention parameters,
    positional embedding/RoPE configuration, token IDs, and initialization options.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int

    layer_types: List[str]
    hidden_activation: str
    rms_norm_eps: float
    use_cache: bool

    query_pre_attn_scalar: int
    sliding_window: int
    _sliding_window_pattern: int
    rope_theta: float
    rope_local_base_freq: float

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    initializer_range: float
    use_bidirectional_attention: bool

    attention_bias: bool
    attention_dropout: float
    attn_logit_softcapping: Optional[float]
    final_logit_softcapping: Optional[float]
    rope_scaling: Optional[dict]

    _attn_implementation: str

    @classmethod
    def from_pretrained(
        cls, repo_id: str, attn_implementation: str = "eager"
    ) -> "Gemma3TextConfig":
        """
        Load config.json, inject `attn_implementation`, and construct the
        config object.
        """
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["_attn_implementation"] = attn_implementation
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**filtered_data)


class Gemma3PreTrainedModel(nn.Module):
    """Minimal base class that replaces transformers.PreTrainedModel.

    Provides a lightweight base without an external dependency on Hugging Face
    Transformers and supplies common utilities used by Gemma3 model classes.
    """

    config: Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config: Gemma3TextConfig = config

    def _init_weights(self, module):
        """
        Standard weight initialization logic for Gemma models.
        Extracted and reproduced only the parts required for Gemma from the
        Transformers source code.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def post_init(self):
        """
        Call at the end of __init__ to initialize all parameters.
        This is mainly important before training to avoid overwriting loaded
        weights, but it's required as a consistency step prior to calling
        `load_state_dict`.
        """
        self.apply(self._init_weights)

    def tie_weights(self):
        """
        Handle weight sharing between input embeddings and the output layer.
        """
        output_embeddings = getattr(self, "lm_head", None)
        input_embeddings = getattr(self.model, "embed_tokens", None)

        if output_embeddings is not None and input_embeddings is not None:
            output_embeddings.weight = input_embeddings.weight


def create_attention_mask(
    *,
    input_embeds: Float[torch.Tensor, "B Q D"],
    attention_mask: Optional[Int[torch.Tensor, "B Q"]],
    cache_position: Int[torch.Tensor, "Q"],
    sliding_window: Optional[int] = None,
) -> Float[torch.Tensor, "B 1 Q K"]:
    """
    Returns attention mask of shape (B, 1, Q, K)
    """
    B, Q, _ = input_embeds.shape
    device = input_embeds.device
    dtype = input_embeds.dtype
    K = Q
    neg_inf = torch.finfo(dtype).min
    q_pos = cache_position.view(Q, 1)
    k_pos = torch.arange(K, device=device).view(1, K)
    mask = torch.zeros((Q, K), device=device, dtype=dtype)
    mask.masked_fill_(k_pos > q_pos, neg_inf)
    if sliding_window is not None:
        too_far = (q_pos - k_pos) > sliding_window
        mask.masked_fill_(too_far, neg_inf)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, Q, K)
    if attention_mask is not None:
        pad = (attention_mask == 0).to(dtype) * neg_inf
        mask = mask + pad[:, None, None, :]
    return mask


def apply_rotary_pos_emb(
    q: Float[torch.Tensor, "B H S D"],
    k: Float[torch.Tensor, "B H S D"],
    cos: Float[torch.Tensor, "B S D"],
    sin: Float[torch.Tensor, "B S D"],
    unsqueeze_dim=1,
) -> tuple[Float[torch.Tensor, "B H S D"], Float[torch.Tensor, "B H S D"]]:
    """
    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [batch, seq_len, head_dim] or [1, seq_len, head_dim]
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: Float[torch.Tensor, "... D"]) -> Float[torch.Tensor, "... D"]:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    embed_scale: Float[torch.Tensor, ""]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input: Int[torch.Tensor, "B S"]) -> Float[torch.Tensor, "B S D"]:
        out = super().forward(input)
        return out * self.embed_scale.to(self.weight.dtype)


class Gemma3RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) helper with cached cos/sin values.

    Computes and caches cosine and sine positional encodings based on the model
    configuration so they can be quickly applied to query/key tensors.
    """

    dim: int
    max_position_embeddings: int
    base: float
    inv_freq: Float[torch.Tensor, "D"]
    max_seq_len_cached: int
    cos_cached: Float[torch.Tensor, "S D"]
    sin_cached: Float[torch.Tensor, "S D"]

    def __init__(self, config: Gemma3TextConfig, device=None):
        super().__init__()
        self.dim: int = config.head_dim
        self.max_position_embeddings: int = config.max_position_embeddings
        self.base: float = config.rope_theta
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).to(device=device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings,
            device=device,
            dtype=torch.float32,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached: int = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: Float[torch.Tensor, "B S D"],
        position_ids: Optional[Int[torch.Tensor, "B S"]],
    ) -> Tuple[Float[torch.Tensor, "B S D"], Float[torch.Tensor, "B S D"]]:
        seq_len = x.shape[1]
        if position_ids is None:
            cos = self.cos_cached[:seq_len].unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0)
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        return cos, sin


class Gemma3RMSNorm(nn.Module):
    """Root-mean-square layer normalization used in Gemma3.

    Implements RMS normalization as used by Gemma architectures, with a
    learnable scaling parameter and a small epsilon for numerical stability.
    """

    eps: float
    weight: nn.Parameter

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: Float[torch.Tensor, "B S D"]) -> Float[torch.Tensor, "B S D"]:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Float[torch.Tensor, "B S D"]) -> Float[torch.Tensor, "B S D"]:
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma3MLP(nn.Module):
    """Feed-forward network (MLP) block used inside decoder layers.

    Contains gated and projection linear layers and applies the configured
    activation function to produce intermediate and output transformations.
    """

    config: Gemma3TextConfig
    hidden_size: int
    intermediate_size: int
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear
    act_fn: Callable[[Float[torch.Tensor, "B S D"]], Float[torch.Tensor, "B S D"]]

    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config: Gemma3TextConfig = config
        self.hidden_size: int = config.hidden_size
        self.intermediate_size: int = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        ACT2FN = {
            "gelu_pytorch_tanh": functools.partial(
                nn.functional.gelu, approximate="tanh"
            ),
        }
        self.act_fn: Callable[[torch.Tensor], torch.Tensor] = ACT2FN[
            config.hidden_activation
        ]

    def forward(self, x: Float[torch.Tensor, "B S D"]) -> Float[torch.Tensor, "B S D"]:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    is_sliding: bool
    config: Gemma3TextConfig
    layer_idx: int
    head_dim: int
    num_key_value_groups: int
    scaling: float
    attention_dropout: float
    is_causal: bool
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    attn_logit_softcapping: Optional[float]
    sliding_window: Optional[int]
    q_norm: Gemma3RMSNorm
    k_norm: Gemma3RMSNorm

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding: bool = config.layer_types[layer_idx] == "sliding_attention"
        self.config: Gemma3TextConfig = config
        self.layer_idx: int = layer_idx
        self.head_dim: int = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups: int = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling: float = config.query_pre_attn_scalar**-0.5
        self.attention_dropout: float = self.config.attention_dropout
        self.is_causal: bool = not self.config.use_bidirectional_attention

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
        self.attn_logit_softcapping: Optional[float] = (
            self.config.attn_logit_softcapping
        )
        self.sliding_window: Optional[int] = (
            config.sliding_window if self.is_sliding else None
        )

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B S D"],
        position_embeddings: Tuple[
            Float[torch.Tensor, "B S D"], Float[torch.Tensor, "B S D"]
        ],
        attention_mask: Optional[Float[torch.Tensor, "B 1 Q K"]],
    ) -> Tuple[Float[torch.Tensor, "B S D"], Optional[Float[torch.Tensor, "B H Q K"]]]:
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

        ALL_ATTENTION_FUNCTIONS = {
            "sdpa": self.sdpa_attention_forward,
            "eager": self.eager_attention_forward,
        }
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            query_states,
            key_states,
            value_states,
            attention_mask,
            is_causal=self.is_causal,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def sdpa_attention_forward(
        self,
        query: Float[torch.Tensor, "B H Q D"],
        key: Float[torch.Tensor, "B H K D"],
        value: Float[torch.Tensor, "B H K D"],
        attention_mask: Optional[Float[torch.Tensor, "B 1 Q K"]],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        sliding_window: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Float[torch.Tensor, "B H Q D"], None]:
        key = self.repeat_kv(key, self.num_key_value_groups)
        value = self.repeat_kv(value, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=bool(is_causal),
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, None

    def eager_attention_forward(
        self,
        query: Float[torch.Tensor, "B H Q D"],
        key: Float[torch.Tensor, "B H K D"],
        value: Float[torch.Tensor, "B H K D"],
        attention_mask: Optional[Float[torch.Tensor, "B 1 Q K"]],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        sliding_window: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Float[torch.Tensor, "B H Q D"], None]:
        key = self.repeat_kv(key, self.num_key_value_groups)
        value = self.repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if scaling is not None:
            attn_weights = attn_weights * scaling

        if self.attn_logit_softcapping is not None:
            attn_weights = self.attn_logit_softcapping * torch.tanh(
                attn_weights / self.attn_logit_softcapping
            )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, None

    @staticmethod
    def repeat_kv(
        hidden_states: Float[torch.Tensor, "B N K D"], n_rep: int
    ) -> Float[torch.Tensor, "B N_rep K D"]:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Gemma3DecoderLayer(nn.Module):
    """Single decoder layer combining attention, MLP, and normalization.

    Each layer encapsulates self-attention (global or sliding), residual
    connections, and feed-forward (MLP) sublayers with pre/post normalizations.
    """

    config: Gemma3TextConfig
    hidden_size: int
    layer_idx: int
    attention_type: str
    self_attn: Gemma3Attention
    mlp: Gemma3MLP
    input_layernorm: Gemma3RMSNorm
    post_attention_layernorm: Gemma3RMSNorm
    pre_feedforward_layernorm: Gemma3RMSNorm
    post_feedforward_layernorm: Gemma3RMSNorm

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config: Gemma3TextConfig = config
        self.hidden_size: int = config.hidden_size
        self.layer_idx: int = layer_idx
        self.attention_type: str = config.layer_types[layer_idx]
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
        hidden_states: Float[torch.Tensor, "B S D"],
        position_embeddings_global: Tuple[
            Float[torch.Tensor, "B S D"], Float[torch.Tensor, "B S D"]
        ],
        position_embeddings_local: Tuple[
            Float[torch.Tensor, "B S D"], Float[torch.Tensor, "B S D"]
        ],
        attention_mask: Optional[Float[torch.Tensor, "B 1 Q K"]] = None,
    ) -> Float[torch.Tensor, "B S D"]:
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
    """Transformer decoder stack for the Gemma3 text model.

    Builds token embeddings, a sequence of decoder layers, and final
    normalization to produce contextualized hidden states for LM heads.
    """

    config: Gemma3TextConfig
    padding_idx: int
    vocab_size: int
    embed_tokens: Gemma3TextScaledWordEmbedding
    layers: nn.ModuleList
    norm: Gemma3RMSNorm
    rotary_emb: Gemma3RotaryEmbedding
    gradient_checkpointing: bool
    rotary_emb_local: Gemma3RotaryEmbedding

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx: int = config.pad_token_id
        self.vocab_size: int = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing: bool = False
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[Int[torch.Tensor, "B S"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "B S"]] = None,
    ) -> Float[torch.Tensor, "B S D"]:
        inputs_embeds = self.embed_tokens(input_ids)

        cache_position = torch.arange(
            0,
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
        position_ids = cache_position.unsqueeze(0)

        full_attn_mask = create_attention_mask(
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )
        sliding_attn_mask = create_attention_mask(
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            sliding_window=self.config.sliding_window,
        )

        hidden_states = inputs_embeds

        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        for decoder_layer in self.layers:
            decoder_layer = cast(Gemma3DecoderLayer, decoder_layer)
            if decoder_layer.attention_type == "sliding_attention":
                current_mask = sliding_attn_mask
            else:
                current_mask = full_attn_mask

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=current_mask,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma3ForCausalLM(Gemma3PreTrainedModel):
    """Causal language modeling wrapper that adds an LM head.

    Wraps the `Gemma3TextModel` and provides a linear language-modeling head
    to project hidden states to vocabulary logits for causal generation.
    """

    model: Gemma3TextModel
    vocab_size: int
    lm_head: nn.Linear

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model: Gemma3TextModel = Gemma3TextModel(config)
        self.vocab_size: int = config.vocab_size
        self.lm_head: nn.Linear = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[Int[torch.Tensor, "B S"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "B S"]] = None,
    ) -> Float[torch.Tensor, "B S V"]:
        outputs: Float[torch.Tensor, "B S D"] = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(outputs)
        return logits


if __name__ == "__main__":
    from safetensors import torch as safetensors_torch
    import transformers
    from huggingface_hub import hf_hub_download
    import torch.nn.functional as F

    tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-3-270m")
    config = Gemma3TextConfig.from_pretrained("google/gemma-3-270m")
    model = Gemma3ForCausalLM(config)
    weight_path = hf_hub_download(
        repo_id="google/gemma-3-270m", filename="model.safetensors"
    )
    weight = safetensors_torch.load_file(weight_path)
    model.load_state_dict(weight, strict=False)
    model.tie_weights()
    text = "Hello, my dog is cute"
    for step in range(20):
        inputs = tokenizer(text, return_tensors="pt")
        logits = model(**inputs)
        next_logits = logits[:, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, 5, dim=-1)

        print(f"\nStep {step}")
        for p, tid in zip(top_probs[0], top_ids[0]):
            token = tokenizer.decode([tid.item()])
            print(f"  {token!r}: {p.item():.4f}")
        next_id = top_ids[0, 0].unsqueeze(0)
        text += tokenizer.decode(next_id)
        print("Generated:", text)
