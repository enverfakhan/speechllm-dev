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
    audio_adapter_r:   int   = 0          # gated audio adapter bottleneck; 0 = disabled


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


# ── Gated audio adapter ─────────────────────────────────────────────────────────

class AudioLayerAdapter(nn.Module):
    """Gated parallel bottleneck adapter applied only at audio-token positions.

    Adds a small residual branch inside a frozen Llama block that reshapes the
    audio stream while leaving every pretrained parameter untouched (all new
    capacity lives here).  The branch computes

        audio_mask * tanh(gate) * up(gelu(down(norm(x))))

    The branch is an *exact* no-op at step 0 — inserting the module never
    perturbs the pretrained forward pass — because ``up_proj`` (the projection
    that writes the residual) is zero-initialised.  The scalar ``gate`` instead
    starts *open* at 1.0 (``tanh(1) ≈ 0.762``), so it is a learned magnitude the
    model can shrink or grow rather than a switch it must first turn on; the
    per-layer ``tanh(gate)`` trace stays the per-depth diagnostic we watch.
    ``norm`` is a fresh RMSNorm owned here (never the block's pretrained norms),
    so no pretrained parameter is read or written.

    Only ``up_proj`` is zeroed: zeroing ``down_proj`` too would make every
    gradient in the branch exactly zero (``d/d down_proj ∝ up_proj.weight`` and
    ``d/d up_proj ∝ gelu(down_proj(·))``), leaving the adapter permanently dead.
    """

    def __init__(self, d_model: int, r: int, eps: float) -> None:
        """Build the bottleneck, its own RMSNorm, and the open scalar gate.

        Args:
            d_model: residual-stream width (matches the block)
            r:       bottleneck rank
            eps:     RMSNorm epsilon (same value the block's norms use)
        """
        super().__init__()
        self.norm      = RMSNorm(d_model, eps)
        self.down_proj = nn.Linear(d_model, r,       bias=False)   # d_model → r
        self.up_proj   = nn.Linear(r,       d_model, bias=False)   # r → d_model (writes residual)
        # Scalar gate, initialised open at 1.0 → tanh(1) ≈ 0.762 (see class docstring).
        self.gate      = nn.Parameter(torch.ones(1))

        # up_proj writes into the residual stream and is zero-initialised, which
        # is what makes the whole branch an exact no-op at step 0.  It is the ONLY
        # tensor here that may be zeroed — down_proj keeps the default nn.Linear
        # init so gradients can flow on the very first step.
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """Return the masked, gated adapter branch (residual add stays in the block).

        Args:
            x:          (B, S, d_model) — the block's INPUT residual stream (the
                        branch runs parallel to the whole layer)
            audio_mask: (B, S, 1) float — 1.0 at audio positions, else 0.0

        Returns:
            (B, S, d_model) — zero everywhere except audio positions, scaled by
            tanh(gate); add this to x in the caller.
        """
        branch = self.up_proj(F.gelu(self.down_proj(self.norm(x))))
        return audio_mask * torch.tanh(self.gate) * branch


# ── Transformer block ─────────────────────────────────────────────────────────

class LlamaBlock(nn.Module):
    """Single Llama transformer block with Pre-RMSNorm and residual connections.

    When ``has_audio_adapter`` and ``config.audio_adapter_r > 0``, the block also
    owns an :class:`AudioLayerAdapter` whose gated branch is added at audio-token
    positions only.  The adapter is the sole new-parameter path; every other
    submodule here is a pretrained Llama parameter and is never modified.
    """

    def __init__(self, config: LlamaConfig, has_audio_adapter: bool = False) -> None:
        """Initialise attention, MLP, their preceding layer norms, and the adapter.

        Args:
            config:            model hyperparameters
            has_audio_adapter: construct the audio adapter for this block (the
                               caller passes False for the last layer, whose
                               audio-position outputs feed nothing)
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

        # Named `audio_adapter` so its parameter names contain that substring —
        # stage setup, grad metrics, and checkpoint tolerance all key off it.
        if has_audio_adapter and config.audio_adapter_r > 0:
            self.audio_adapter: AudioLayerAdapter | None = AudioLayerAdapter(
                config.d_model, config.audio_adapter_r, config.rms_norm_eps
            )
        else:
            self.audio_adapter = None

    def forward(
        self,
        x:          torch.Tensor,
        cos:        torch.Tensor,
        sin:        torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention, MLP, and (optionally) the gated audio adapter.

        Args:
            x:          (B, S, d_model)
            cos:        (S, head_dim)
            sin:        (S, head_dim)
            audio_mask: (B, S, 1) float mask, or None to disable the adapter
                        branch (text-only / stub paths)

        Returns:
            (B, S, d_model)
        """
        # Read the BLOCK INPUT: the adapter runs parallel to the whole layer
        # (attention + MLP), not just parallel to the MLP.
        if self.audio_adapter is not None and audio_mask is not None:
            x_hat = self.audio_adapter(x, audio_mask)
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        # Zero-init up_proj makes this an exact no-op at step 0; still skip the
        # compute when there is no adapter or no mask (frozen text path untouched).
        if self.audio_adapter is not None and audio_mask is not None:
            x = x + x_hat
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
        # Skip the audio adapter on the LAST layer: its audio-position outputs
        # feed nothing (only transcript positions read the final hidden state).
        self.layers       = nn.ModuleList(
            [LlamaBlock(config, has_audio_adapter=(i < config.n_layers - 1))
             for i in range(config.n_layers)]
        )
        self._has_audio_adapters = any(
            layer.audio_adapter is not None for layer in self.layers
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
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run a forward pass and optionally compute next-token-prediction loss.

        Args:
            inputs_embeds: (B, S, d_model) — full embedded sequence from prepare_input()
            labels:        (B, S) — -100 at masked positions; true token IDs at transcript
            audio_lengths: (B,) — per-sample audio-token count; audio occupies
                           positions [0, audio_lengths[i]).  When provided (and the
                           model has audio adapters) it builds the audio mask so the
                           gated adapters fire only at audio positions.  When None,
                           the adapters stay inactive — text-only and stub self-test
                           paths behave exactly as before.

        Returns:
            logits: (B, S, vocab_size)
            loss:   scalar cross-entropy averaged over unmasked positions, or None
        """
        _, S, _ = inputs_embeds.shape

        cos = self.rope_cos[:S]   # (S, head_dim)
        sin = self.rope_sin[:S]

        # Build the audio mask once and share it across every layer.  Masking the
        # full-sequence branch output is the legible choice: audio lengths vary
        # across the batch, and audio dominates sequence length anyway, so
        # per-sample slicing would buy nothing but complexity.
        audio_mask: torch.Tensor | None = None
        if audio_lengths is not None and self._has_audio_adapters:
            positions  = torch.arange(S, device=inputs_embeds.device)
            audio_mask = (
                (positions[None, :] < audio_lengths[:, None])   # (B, S) bool
                .unsqueeze(-1)                                    # (B, S, 1)
                .to(inputs_embeds.dtype)
            )

        x = inputs_embeds
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cos, sin, audio_mask, use_reentrant=False)
            else:
                x = layer(x, cos, sin, audio_mask)

        x      = self.norm(x)
        logits = self.lm_head(x)   # (B, S, vocab_size)

        loss = None
        if labels is not None:
            # Shift by one: logits[i] predicts labels[i+1]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation recomputation during backward to reduce peak VRAM."""
        self.gradient_checkpointing = True

    def audio_adapter_parameters(self) -> list[nn.Parameter]:
        """Return every parameter belonging to the gated audio adapters.

        Selected by the substring 'audio_adapter' in the parameter name — the
        same predicate stage setup and checkpoint tolerance use, so the three
        stay in lockstep.  Empty when audio adapters are disabled.
        """
        return [p for name, p in self.named_parameters() if "audio_adapter" in name]

    def audio_gate_values(self) -> dict[str, float]:
        """Return per-layer effective gate values tanh(gate) for W&B logging.

        This is the key diagnostic for the gated-adapter experiment: which
        depths open their gates, and how fast.  Keys are 'gates/layer_XX'.
        """
        return {
            f"gates/layer_{i:02d}": float(torch.tanh(layer.audio_adapter.gate).item())
            for i, layer in enumerate(self.layers)
            if layer.audio_adapter is not None
        }

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


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    torch.manual_seed(0)

    # Tiny CPU-only config; audio_adapter_r > 0 constructs the gated adapters.
    _COMMON = dict(
        n_layers=3, d_model=64, n_heads=4, n_kv_heads=2,
        intermediate_size=128, vocab_size=100,
    )
    cfg_on  = LlamaConfig(**_COMMON, audio_adapter_r=16)
    cfg_off = LlamaConfig(**_COMMON, audio_adapter_r=0)

    model_on  = Llama(cfg_on)
    model_off = Llama(cfg_off)

    # Make the two models share every pretrained-path weight so a forward-pass
    # comparison isolates the adapter branch.  model_on's extra audio_adapter.*
    # keys are the only ones that should be "unexpected" for model_off.
    incompat = model_off.load_state_dict(model_on.state_dict(), strict=False)
    assert not incompat.missing_keys, f"unexpected missing keys: {incompat.missing_keys}"
    assert all("audio_adapter" in k for k in incompat.unexpected_keys), incompat.unexpected_keys

    model_on.eval()
    model_off.eval()

    B, S, D = 2, 10, cfg_on.d_model
    inputs        = torch.randn(B, S, D)
    audio_lengths = torch.tensor([4, 7])

    # ── Test: last layer has no adapter ────────────────────────────────────────
    assert model_on.layers[-1].audio_adapter is None, "last layer must not own an adapter"
    assert all(model_on.layers[i].audio_adapter is not None for i in range(cfg_on.n_layers - 1))
    assert model_off.layers[0].audio_adapter is None, "r=0 must construct no adapters"
    print("[OK] last layer has no adapter; r=0 constructs none")

    # ── Test: identity at init (up_proj == 0 → exact no-op) ─────────────────────
    with torch.no_grad():
        logits_on,  _ = model_on(inputs, audio_lengths=audio_lengths)
        logits_off, _ = model_off(inputs, audio_lengths=audio_lengths)
    assert torch.allclose(logits_on, logits_off, atol=1e-6), (
        f"adapter branch must be a no-op at init; max diff "
        f"{(logits_on - logits_off).abs().max().item():.2e}"
    )
    print("[OK] identity at init: r=16 (up_proj=0) matches r=0")

    # ── Test: mask correctness at a single block ───────────────────────────────
    # Comparing whole-model logits would let causal attention spread an
    # audio-position perturbation to transcript positions.  A single block adds
    # the masked branch AFTER attention/MLP, so it perturbs ONLY audio positions
    # — which is exactly what the mask claims to guarantee.
    block = LlamaBlock(cfg_on, has_audio_adapter=True)
    block.eval()
    head_dim = cfg_on.d_model // cfg_on.n_heads
    cos, sin = _precompute_rope_cos_sin(head_dim, S, cfg_on.rope_theta)
    x = torch.randn(B, S, D)
    audio_mask = (
        (torch.arange(S)[None, :] < audio_lengths[:, None]).unsqueeze(-1).float()
    )
    with torch.no_grad():
        ref = block(x, cos, sin, audio_mask=None)          # adapter disabled
        # up_proj is zero at init, so give it real weights — otherwise the branch
        # is still the exact no-op the previous test just checked.
        block.audio_adapter.up_proj.weight.normal_(mean=0.0, std=0.05)
        out = block(x, cos, sin, audio_mask)               # adapter active at audio pos
    for i in range(B):
        n = int(audio_lengths[i].item())
        assert not torch.allclose(out[i, :n], ref[i, :n]), (
            f"sample {i}: audio positions [0,{n}) should change when gate opens"
        )
        assert torch.allclose(out[i, n:], ref[i, n:], atol=1e-6), (
            f"sample {i}: non-audio positions [{n},{S}) must be untouched"
        )
    print("[OK] mask correctness: only audio positions change when a gate opens")

    # ── Test: gradient-checkpointing path, adapter-only training regime ─────────
    # Mirrors the real stage: pretrained params frozen, inputs_embeds require no
    # grad (frozen adapter/embed), only the audio adapters train.  use_reentrant
    # =False must still route gradients into the adapters.
    gc_model = Llama(cfg_on)
    gc_model.train()
    gc_model.enable_gradient_checkpointing()
    for name, p in gc_model.named_parameters():
        p.requires_grad_("audio_adapter" in name)
    for layer in gc_model.layers:
        if layer.audio_adapter is not None:
            # Lift up_proj off its zero init so down_proj, norm and gate — whose
            # grads are all proportional to it — see signal on this first step.
            layer.audio_adapter.up_proj.weight.data.normal_(mean=0.0, std=0.05)

    gc_inputs = torch.randn(B, S, D)                        # requires_grad=False (frozen upstream)
    gc_labels = torch.full((B, S), -100, dtype=torch.long)
    gc_labels[:, 7:] = torch.randint(0, cfg_on.vocab_size, (B, S - 7))
    _, gc_loss = gc_model(gc_inputs, gc_labels, audio_lengths=audio_lengths)
    gc_loss.backward()

    aa_params = gc_model.audio_adapter_parameters()
    assert aa_params, "expected non-empty audio adapter parameter list"
    assert all(p.grad is not None for p in aa_params), "every audio adapter param needs a grad"
    assert all(p.grad.abs().sum().item() > 0 for p in aa_params), (
        "every audio adapter param (norm, down, up, gate) must receive a non-zero grad"
    )
    for name, p in gc_model.named_parameters():
        if "audio_adapter" not in name:
            assert not p.requires_grad, f"pretrained param {name} should be frozen"
            assert p.grad is None, f"frozen param {name} should have no grad"
    print("[OK] gradient checkpointing: adapters get grads, frozen llama stays grad-free")

    # ── Test: helper methods ───────────────────────────────────────────────────
    gates = model_on.audio_gate_values()
    assert set(gates.keys()) == {f"gates/layer_{i:02d}" for i in range(cfg_on.n_layers - 1)}, gates
    assert all(abs(v - math.tanh(1.0)) < 1e-6 for v in gates.values()), (
        f"gates should start open at tanh(1)={math.tanh(1.0):.4f}: {gates}"
    )
    assert model_off.audio_gate_values() == {}, "r=0 model exposes no gates"
    print("[OK] audio_gate_values / audio_adapter_parameters helpers")

    print("\nPASSED")
    sys.exit(0)
