"""Llama 3.1 8B, implemented from scratch.

Zero dependency on `transformers`. All subcomponents implemented explicitly in
this single file: RMSNorm, RoPE, GQA attention (n_heads=32, n_kv_heads=8),
SwiGLU MLP, transformer blocks (Pre-RMSNorm), LM head with tied embeddings.

The model always takes pre-embedded inputs (inputs_embeds) — embedding happens
in prepare_input() in adapter.py, which also owns the token embedding layer.

Weight loading supports two on-disk formats:
  HuggingFace safetensors  model.safetensors.index.json + shard files
  Meta native .pth         consolidated.00.pth (one or more shards)
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from linear_cross_entropy import linear_cross_entropy as _fused_lce


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class LlamaConfig:
    """Llama 3.1 8B architecture hyperparameters."""

    n_layers:          int   = 32
    d_model:           int   = 4096
    n_heads:           int   = 32
    n_kv_heads:        int   = 8          # GQA: fewer KV heads than Q heads
    intermediate_size: int   = 14336
    vocab_size:        int   = 32000      # overridden after build_vocab.py
    max_seq_len:       int   = 131072
    rms_norm_eps:      float = 1e-5
    rope_theta:        float = 500000.0   # Llama 3.1 uses 500k (3.0 used 10k)


# ── RMSNorm ────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean-centering, no bias)."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        """Initialise scale weights to ones.

        Args:
            d_model: feature dimension to normalise over
            eps:     stability epsilon added before taking the square root
        """
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x and apply learned per-channel scale.

        Args:
            x: (..., d_model)

        Returns:
            (..., d_model)
        """
        # rsqrt = 1 / sqrt(mean(x^2) + eps)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm)


# ── RoPE helpers ───────────────────────────────────────────────────────────────

def _precompute_rope_cos_sin(
    head_dim: int,
    max_seq_len: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables for all positions 0 … max_seq_len-1.

    Args:
        head_dim:    per-head feature dimension (d_model // n_heads)
        max_seq_len: maximum sequence length supported
        theta:       base frequency (500 000 for Llama 3.1)

    Returns:
        cos: (max_seq_len, head_dim) float32
        sin: (max_seq_len, head_dim) float32
    """
    # head_dim // 2 distinct frequencies, one per dimension pair
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles    = torch.outer(positions, inv_freq)   # (max_seq_len, head_dim // 2)
    # Duplicate to cover both halves used by the rotate-half trick
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)  # (max_seq_len, head_dim)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate each vector: [x1 | x2] → [-x2 | x1]."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(
    q:   torch.Tensor,
    k:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q:   (B, n_heads,    S, head_dim)
        k:   (B, n_kv_heads, S, head_dim)
        cos: (S, head_dim) — sliced from precomputed table
        sin: (S, head_dim)

    Returns:
        Rotated (q, k) with unchanged shapes.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, S, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q   = q * cos + _rotate_half(q) * sin
    k   = k * cos + _rotate_half(k) * sin
    return q, k


# ── Attention ──────────────────────────────────────────────────────────────────

class GQAAttention(nn.Module):
    """Grouped-Query Attention with causal masking and RoPE.

    n_heads Q heads share n_kv_heads K/V heads in groups of
    n_heads // n_kv_heads.  Before the attention kernel the K and V tensors are
    expanded with repeat_interleave so every Q head has a matching K/V head.
    All projections are bias-free (matches Meta's implementation).
    """

    def __init__(self, config: LlamaConfig) -> None:
        """Initialise Q, K, V, O projections from config.

        Args:
            config: model hyperparameters
        """
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim   = config.d_model // config.n_heads
        self.n_groups   = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model,                    bias=False)

    def forward(
        self,
        x:   torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-head GQA with RoPE and causal mask.

        Args:
            x:   (B, S, d_model)
            cos: (S, head_dim) — from precomputed RoPE table
            sin: (S, head_dim)

        Returns:
            (B, S, d_model)
        """
        B, S, _ = x.shape

        q = self.q_proj(x)   # (B, S, n_heads    * head_dim)
        k = self.k_proj(x)   # (B, S, n_kv_heads * head_dim)
        v = self.v_proj(x)   # (B, S, n_kv_heads * head_dim)

        q = q.reshape(B, S, self.n_heads,    self.head_dim).transpose(1, 2)
        k = k.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        # Expand KV heads to match Q head count for GQA
        # (B, n_kv_heads, S, head_dim) → (B, n_heads, S, head_dim)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # FlashAttention-2 used automatically when available
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)

        return self.o_proj(out)


# ── MLP ────────────────────────────────────────────────────────────────────────

class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward: down_proj(silu(gate_proj(x)) ⊙ up_proj(x)).

    gate_proj and up_proj run in parallel; their element-wise product
    (gated by SiLU) is projected back to d_model by down_proj.
    All projections are bias-free.
    """

    def __init__(self, config: LlamaConfig) -> None:
        """Initialise gate, up, and down projections.

        Args:
            config: model hyperparameters
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args/returns: (B, S, d_model)."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer block ─────────────────────────────────────────────────────────

class LlamaBlock(nn.Module):
    """Single Llama transformer block with Pre-RMSNorm and residual connections."""

    def __init__(self, config: LlamaConfig) -> None:
        """Initialise attention, MLP, and their preceding layer norms.

        Args:
            config: model hyperparameters
        """
        super().__init__()
        self.input_layernorm          = RMSNorm(config.d_model, config.rms_norm_eps)
        self.self_attn                = GQAAttention(config)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp                      = SwiGLUMLP(config)

        # Scale residual projections so signal magnitude doesn't compound across layers.
        # This is the "scaled init" from the GPT-2 paper: std = 0.02 / sqrt(n_layers).
        # Applied to the two projections that write into the residual stream.
        std = 0.02 / math.sqrt(config.n_layers)
        nn.init.normal_(self.self_attn.o_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.mlp.down_proj.weight,    mean=0.0, std=std)

    def forward(
        self,
        x:   torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention and MLP sub-layers with residual connections.

        Args:
            x:   (B, S, d_model)
            cos: (S, head_dim)
            sin: (S, head_dim)

        Returns:
            (B, S, d_model)
        """
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ── Full model ─────────────────────────────────────────────────────────────────

class Llama(nn.Module):
    """Llama 3.1 decoder-only transformer, accepting pre-embedded inputs.

    Never takes raw token IDs — the embedding step lives in prepare_input()
    (adapter.py) so the embedding table is accessible there for constructing
    the full sequence.  The LM head shares the embedding weight (tied).
    """

    def __init__(self, config: LlamaConfig) -> None:
        """Construct all layers and precompute RoPE buffers.

        Args:
            config: model hyperparameters
        """
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers       = nn.ModuleList(
            [LlamaBlock(config) for _ in range(config.n_layers)]
        )
        self.norm    = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie LM head to the token embedding — they are the same parameter object.
        # named_parameters() deduplicates, so only embed_tokens.weight is trained;
        # lm_head.weight is an alias, not a separate parameter.
        self.lm_head.weight = self.embed_tokens.weight

        # Initialise embed_tokens (and tied lm_head) with small std so initial
        # logits are near-zero → output distribution starts near-uniform →
        # loss starts near log(vocab_size) ≈ 10.6 instead of 300+.
        # Without this, random matrix multiplications across N layers compound
        # to produce logits with magnitude ~50-200, causing near-one-hot
        # predictions on arbitrary tokens and 300+ steps of saturation escape
        # before any real learning happens.
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02 / math.sqrt(config.n_layers))

        # Precomputed RoPE tables — stored as non-trainable buffers so they move
        # with the model when calling .to(device) or .cuda()
        head_dim = config.d_model // config.n_heads
        cos, sin = _precompute_rope_cos_sin(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        labels:        torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run a forward pass and optionally compute next-token-prediction loss.

        Args:
            inputs_embeds: (B, S, d_model) — full embedded sequence from prepare_input()
            labels:        (B, S) — -100 at masked positions; true token IDs at transcript

        Returns:
            logits: (B, S, vocab_size)
            loss:   scalar cross-entropy averaged over unmasked positions, or None
        """
        _, S, _ = inputs_embeds.shape

        cos = self.rope_cos[:S]   # (S, head_dim)
        sin = self.rope_sin[:S]

        x = inputs_embeds
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)

        x = self.norm(x)

        # ── Inference: materialise full logits ────────────────────────────────
        if labels is None:
            return self.lm_head(x), None

        # ── Training: fused projection + loss, transcript positions only ──────
        # Causal shift: hidden[i] predicts labels[i+1].
        shift_hidden = x[:, :-1, :].contiguous()       # (B, S-1, d_model)
        shift_labels = labels[:, 1:].contiguous()       # (B, S-1)

        flat_hidden = shift_hidden.view(-1, self.config.d_model)
        flat_labels = shift_labels.view(-1)

        valid = flat_labels != -100
        h = flat_hidden[valid]   # (N_transcript, d_model)
        t = flat_labels[valid]   # (N_transcript,)

        # _fused_lce requires N % 512 == 0 and V % 4096 == 0.
        N_pad = (512 - h.size(0) % 512) % 512
        if N_pad:
            h = F.pad(h, (0, 0, 0, N_pad))
            t = F.pad(t, (0, N_pad), value=-100)

        V_pad = (4096 - self.config.vocab_size % 4096) % 4096
        At = self.lm_head.weight.t()        # (d_model, vocab_size)
        if V_pad:
            At = F.pad(At, (0, V_pad))      # (d_model, vocab_size_padded)

        # N_chunk_size=5 caps the logit buffer at 5 × vocab_size rows at once,
        # keeping the peak kernel allocation small instead of the default 4096.
        loss, *_ = _fused_lce(h, t, At, N_chunk_size=5)

        return None, loss

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation recomputation during backward to reduce peak VRAM."""
        self.gradient_checkpointing = True

    def load_meta_weights(
        self,
        checkpoint_dir: Path,
        vocab_map: dict[str, int] | None = None,
    ) -> None:
        """Load pretrained Llama 3.1 weights from a local checkpoint directory.

        Supports two on-disk formats:
          HuggingFace safetensors  model.safetensors.index.json + shard .safetensors files
          Meta native .pth         consolidated.00.pth (one or more numbered shards)

        Asserts on every parameter shape before copying. Raises immediately on
        any mismatch so shape bugs are never silent.

        When the model is initialised with a pruned vocabulary (vocab_size < full
        checkpoint vocab size), the embedding matrix row counts will differ.
        Pass vocab_map (the dict from vocab_map.json, mapping old_id_str → new_id)
        to extract the relevant rows from the checkpoint and initialise the pruned
        embedding from pretrained weights.  Without vocab_map the embedding is skipped
        and left at its random initialisation.

        Args:
            checkpoint_dir: directory containing the checkpoint files
            vocab_map:      optional {str(old_token_id): new_token_id} mapping produced
                            by build_vocab.py; enables pretrained embedding init for
                            pruned vocabularies
        """
        checkpoint_dir = Path(checkpoint_dir)

        # ── Collect flat state dict from disk ─────────────────────────────────
        state: dict[str, torch.Tensor] = {}
        fmt: str

        hf_index  = checkpoint_dir / "model.safetensors.index.json"
        hf_single = checkpoint_dir / "model.safetensors"
        meta_glob = sorted(checkpoint_dir.glob("consolidated.*.pth"))

        if hf_index.exists():
            from safetensors.torch import load_file
            with open(hf_index) as f:
                index = json.load(f)
            for shard in sorted(set(index["weight_map"].values())):
                state.update(load_file(checkpoint_dir / shard, device="cpu"))
            fmt = "hf"
        elif hf_single.exists():
            from safetensors.torch import load_file
            state = load_file(hf_single, device="cpu")
            fmt = "hf"
        elif meta_glob:
            for shard in meta_glob:
                state.update(
                    torch.load(shard, map_location="cpu", weights_only=True)
                )
            fmt = "meta"
        else:
            raise FileNotFoundError(
                f"No recognised checkpoint found in '{checkpoint_dir}'. "
                "Expected model.safetensors.index.json, model.safetensors, "
                "or consolidated.*.pth."
            )

        # ── Map checkpoint key names to our parameter names ───────────────────
        def _remap(key: str) -> str | None:
            """Return our parameter name for a checkpoint key, or None to skip."""
            if fmt == "hf":
                # lm_head.weight is tied to embed_tokens.weight; skip to avoid
                # a duplicate copy_ (embed_tokens.weight is already loaded).
                if key == "lm_head.weight":
                    return None
                if key.startswith("model."):
                    return key[len("model."):]   # strip "model." prefix
                return None  # unknown top-level key

            # Meta native key → our key
            _top = {
                "tok_embeddings.weight": "embed_tokens.weight",
                "norm.weight":           "norm.weight",
                # output.weight is tied to tok_embeddings; skip for the same reason.
            }
            if key in _top:
                return _top[key]

            _layer = {
                "attention_norm.weight":   "input_layernorm.weight",
                "ffn_norm.weight":         "post_attention_layernorm.weight",
                "attention.wq.weight":     "self_attn.q_proj.weight",
                "attention.wk.weight":     "self_attn.k_proj.weight",
                "attention.wv.weight":     "self_attn.v_proj.weight",
                "attention.wo.weight":     "self_attn.o_proj.weight",
                "feed_forward.w1.weight":  "mlp.gate_proj.weight",
                "feed_forward.w2.weight":  "mlp.down_proj.weight",
                "feed_forward.w3.weight":  "mlp.up_proj.weight",
            }
            # layers.N.<suffix> → layers.N.<remapped_suffix>
            for i in range(self.config.n_layers):
                pfx = f"layers.{i}."
                if key.startswith(pfx) and key[len(pfx):] in _layer:
                    return pfx + _layer[key[len(pfx):]]

            return None  # skip (e.g. rope.freqs, output.weight)

        # ── Load with shape assertion ─────────────────────────────────────────
        # named_parameters() deduplicates tied weights: lm_head.weight will not
        # appear here (embed_tokens.weight is the canonical entry).
        own_params = dict(self.named_parameters())

        loaded, skipped = 0, 0
        for ckpt_key, tensor in state.items():
            our_key = _remap(ckpt_key)
            if our_key is None or our_key not in own_params:
                skipped += 1
                continue
            param = own_params[our_key]
            if tensor.shape != param.shape:
                if our_key == "embed_tokens.weight":
                    if vocab_map is not None:
                        # Extract the rows for pruned tokens from the full checkpoint embedding.
                        # vocab_map maps str(old_id) → new_id; invert to new_id → old_id index list.
                        new_to_old = [0] * param.shape[0]
                        for old_str, new_id in vocab_map.items():
                            new_to_old[new_id] = int(old_str)
                        pruned = tensor[new_to_old, :]
                        assert pruned.shape == param.shape, (
                            f"Pruned embedding shape {tuple(pruned.shape)} != "
                            f"model shape {tuple(param.shape)}"
                        )
                        with torch.no_grad():
                            param.copy_(pruned)
                        loaded += 1
                        print(
                            f"[load_meta_weights] embed_tokens.weight initialised from "
                            f"pretrained rows (checkpoint {tuple(tensor.shape)} → "
                            f"pruned {tuple(pruned.shape)})"
                        )
                        continue
                    print(
                        f"[load_meta_weights] skipping embed_tokens.weight "
                        f"(checkpoint {tuple(tensor.shape)} vs model {tuple(param.shape)})"
                        " — pass vocab_map to initialise from pretrained rows"
                    )
                    skipped += 1
                    continue
                raise ValueError(
                    f"Shape mismatch '{ckpt_key}' → '{our_key}': "
                    f"checkpoint {tuple(tensor.shape)} vs model {tuple(param.shape)}"
                )
            with torch.no_grad():
                param.copy_(tensor)
            loaded += 1

        if loaded == 0:
            raise RuntimeError(
                f"No parameters were loaded from '{checkpoint_dir}'. "
                "Check that the checkpoint format is supported and the directory "
                "contains the expected files."
            )
