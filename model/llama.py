"""Llama 3.1 8B, implemented from scratch.

Zero dependency on `transformers`. All subcomponents implemented explicitly in
this single file: RMSNorm, RoPE, GQA attention (n_heads=32, n_kv_heads=8),
SwiGLU MLP, transformer blocks (Pre-RMSNorm), LM head with tied embeddings.

The model always takes pre-embedded inputs (inputs_embeds) — sequence assembly
(model/sequence.py) embeds the scaffold/instruction/transcript tokens through
this model's embed_tokens and splices the bridge's audio embeddings in.

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
    audio_adapter_r:   int   = 0          # per-layer SwiGLU audio adapter width; 0 = disabled


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


# ── Audio adapters ─────────────────────────────────────────────────────────────

class AudioSwiGLUAdapter(nn.Module):
    """Parallel SwiGLU adapter applied only at audio-token positions.

    Adds a small residual branch inside a frozen Llama block that reshapes the
    audio stream while leaving every pretrained parameter untouched (all new
    capacity lives here).  The branch computes

        audio_mask * down_proj(silu(gate_proj(h)) ⊙ up_proj(h))
        h = norm(x)

    ``norm`` is a fresh RMSNorm owned here (never the block's pretrained norms).
    Projection names follow :class:`SwiGLUMLP`: ``gate_proj``/``up_proj`` are
    the two d_model→r input projections and ``down_proj`` (r→d_model) writes
    the residual.  ``gate_proj`` is the branch's gate — per-channel and
    input-dependent, which is why there is no additional scalar gate.

    Init: ``down_proj`` gets the GPT-2 scaled init (std = 0.02/sqrt(L)) that
    o_proj and mlp.down_proj use, so once the branch turns on it writes at the
    same magnitude as a regular transformer layer.  The exact no-op at step 0
    comes from zeroing ``gate_proj``: ``silu(0) == 0`` kills the product, while
    ``silu'(0) == 0.5`` keeps its gradient alive, so ``gate_proj`` trains on
    step 0 and the rest join from step 1.  It is the ONLY zeroed tensor —
    zeroing a second projection makes every gradient in the branch identically
    zero and the adapter never trains.
    """

    def __init__(self, d_model: int, r: int, n_layers: int, eps: float) -> None:
        """Build the SwiGLU bottleneck and its own RMSNorm.

        Args:
            d_model:  residual-stream width (matches the block)
            r:        bottleneck width (the SwiGLU intermediate size)
            n_layers: model depth; sets the GPT-2 scaled init std for down_proj
            eps:      RMSNorm epsilon (same value the block's norms use)
        """
        super().__init__()
        self.norm      = RMSNorm(d_model, eps)
        self.gate_proj = nn.Linear(d_model, r,       bias=False)   # d_model → r (SwiGLU gate)
        self.up_proj   = nn.Linear(d_model, r,       bias=False)   # d_model → r (SwiGLU value)
        self.down_proj = nn.Linear(r,       d_model, bias=False)   # r → d_model (writes residual)

        # Writer at transformer-layer scale (same std as o_proj / mlp.down_proj)…
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02 / math.sqrt(n_layers))
        # …so the no-op comes from silu(0) = 0 instead.  Gradient still flows
        # here because silu'(0) = 0.5 (see class docstring).
        nn.init.zeros_(self.gate_proj.weight)

    def scales(self) -> dict[str, float]:
        """Per-layer magnitude diagnostic: RMS of the writer and gate weights."""
        return {
            "writer": float(self.down_proj.weight.detach().pow(2).mean().sqrt().item()),
            "input":  float(self.gate_proj.weight.detach().pow(2).mean().sqrt().item()),
        }

    def forward(self, x: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """Return the masked adapter branch (residual add stays in the block).

        Args:
            x:          (B, S, d_model) — the block's INPUT residual stream
            audio_mask: (B, S, 1) float — 1.0 at audio positions, else 0.0

        Returns:
            (B, S, d_model) — zero everywhere except audio positions; add this
            to x in the caller.
        """
        h      = self.norm(x)
        branch = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return audio_mask * branch


# ── Transformer block ─────────────────────────────────────────────────────────

class LlamaBlock(nn.Module):
    """Single Llama transformer block with Pre-RMSNorm and residual connections.

    When ``has_audio_adapter`` and ``config.audio_adapter_r > 0``, the block also
    owns an :class:`AudioSwiGLUAdapter` whose branch is added at audio-token
    positions only.  The adapter is the sole
    new-parameter path; every other submodule here is a pretrained Llama
    parameter and is never modified.
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
            self.audio_adapter: AudioSwiGLUAdapter | None = AudioSwiGLUAdapter(
                config.d_model, config.audio_adapter_r, config.n_layers, config.rms_norm_eps,
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
        """Apply attention, MLP, and (optionally) the audio adapter branch.

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
        # The adapter's zero-initialised gate_proj makes this an exact no-op at
        # step 0; still skip the compute when there is no adapter or no mask
        # (keeps the frozen text path untouched).
        if self.audio_adapter is not None and audio_mask is not None:
            x = x + x_hat
        return x


# ── Full model ─────────────────────────────────────────────────────────────────

class Llama(nn.Module):
    """Llama 3.1 decoder-only transformer, accepting pre-embedded inputs.

    Never takes raw token IDs — the embedding step lives in sequence assembly
    (model/sequence.py) so the embedding table is accessible there for
    constructing the full sequence.  The LM head shares the embedding weight (tied).
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
        audio_mask:    torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run a forward pass and optionally compute next-token-prediction loss.

        Args:
            inputs_embeds: (B, S, d_model) — full embedded sequence (model/sequence.py)
            labels:        (B, S) — -100 at masked positions; true token IDs at targets
            audio_mask:    (B, S, 1) — 1.0 at audio-content positions, 0.0 elsewhere,
                           built by whoever assembled the sequence (the audio sits
                           at a scaffold offset this module cannot know).  None
                           leaves the audio adapters inactive.

        Returns:
            logits: (B, S, vocab_size)
            loss:   scalar cross-entropy averaged over unmasked positions, or None
        """
        _, S, _ = inputs_embeds.shape

        cos = self.rope_cos[:S]   # (S, head_dim)
        sin = self.rope_sin[:S]

        # One mask shared by every layer.
        if audio_mask is not None:
            if audio_mask.shape != (inputs_embeds.shape[0], S, 1):
                raise ValueError(
                    f"audio_mask shape {tuple(audio_mask.shape)} != expected "
                    f"{(inputs_embeds.shape[0], S, 1)}"
                )
            audio_mask = audio_mask.to(inputs_embeds.dtype)

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
        """Return every parameter belonging to the audio adapters.

        Selected by the substring 'audio_adapter' in the parameter name — the
        same predicate stage setup and checkpoint tolerance use, so the three
        stay in lockstep.  Empty when audio adapters are disabled.
        """
        return [p for name, p in self.named_parameters() if "audio_adapter" in name]

    def audio_adapter_scales(self) -> dict[str, float]:
        """Return per-layer adapter weight magnitudes for W&B logging.

        Replaces the old tanh(gate) trace (the scalar gate is gone) as the
        which-depths-engage diagnostic.  Two scalars per layer, keys
        'adapter_scale/layer_XX_writer' and '…_input':

          writer  RMS of down_proj, the projection that writes the residual
          input   RMS of gate_proj, the zero-initialised gate

        The input trace starts at exactly 0, so its rise is the "this depth is
        engaging" signal; the writer starts at its init scale and moving there
        means the branch is being reshaped.
        """
        return {
            f"adapter_scale/layer_{i:02d}_{key}": value
            for i, layer in enumerate(self.layers)
            if layer.audio_adapter is not None
            for key, value in layer.audio_adapter.scales().items()
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
        loaded_keys: set[str] = set()
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
                        loaded_keys.add(our_key)
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
            loaded_keys.add(our_key)

        if loaded == 0:
            raise RuntimeError(
                f"No parameters were loaded from '{checkpoint_dir}'. "
                "Check that the checkpoint format is supported and the directory "
                "contains the expected files."
            )

        # A caller that supplied vocab_map asked for a pretrained embedding.  If
        # the checkpoint's key naming meant we never reached embed_tokens, the
        # model would silently keep its random rows — including the chat
        # specials, whose pretrained values are the reason they were forced into
        # the vocabulary at all.  Fail instead.
        if vocab_map is not None and "embed_tokens.weight" not in loaded_keys:
            raise RuntimeError(
                f"vocab_map was supplied but no embedding was loaded from "
                f"'{checkpoint_dir}' — the token embedding would stay randomly "
                "initialised. Check the checkpoint format/key names."
            )


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    torch.manual_seed(0)

    # Tiny CPU-only config; audio_adapter_r > 0 constructs the audio adapters.
    _COMMON = dict(
        n_layers=3, d_model=64, n_heads=4, n_kv_heads=2,
        intermediate_size=128, vocab_size=100,
    )
    cfg_on  = LlamaConfig(**_COMMON, audio_adapter_r=16)
    cfg_off = LlamaConfig(**_COMMON, audio_adapter_r=0)

    B, S, D = 2, 10, cfg_on.d_model

    def _span_mask(start: int, lengths: list[int]) -> torch.Tensor:
        """(B, S, 1) mask that is 1.0 on [start, start + lengths[i]) per sample."""
        pos = torch.arange(S)[None, :]
        lens = torch.tensor(lengths)[:, None]
        return ((pos >= start) & (pos < start + lens)).unsqueeze(-1).float()

    def _lift_gates(model: Llama) -> None:
        """Give every adapter's zero-initialised gate_proj real weights."""
        for layer in model.layers:
            if layer.audio_adapter is not None:
                layer.audio_adapter.gate_proj.weight.data.normal_(mean=0.0, std=0.05)

    inputs = torch.randn(B, S, D)
    mask   = _span_mask(3, [4, 2])     # audio at a scaffold offset, as in the chat layout

    model_on  = Llama(cfg_on)
    model_off = Llama(cfg_off)
    # Share every pretrained-path weight so a forward comparison isolates the
    # adapter branch; the audio_adapter.* keys are the only unexpected ones.
    incompat = model_off.load_state_dict(model_on.state_dict(), strict=False)
    assert not incompat.missing_keys, f"unexpected missing keys: {incompat.missing_keys}"
    assert all("audio_adapter" in k for k in incompat.unexpected_keys), incompat.unexpected_keys
    model_on.eval()
    model_off.eval()

    # ── Test: adapter placement ────────────────────────────────────────────────
    assert model_on.layers[-1].audio_adapter is None, "last layer must not own an adapter"
    assert all(isinstance(model_on.layers[i].audio_adapter, AudioSwiGLUAdapter)
               for i in range(cfg_on.n_layers - 1))
    assert model_off.layers[0].audio_adapter is None, "r=0 must construct no adapters"
    print("[OK] adapters on layers 0..n-2; last layer has none; r=0 constructs none")

    # ── Test: init — gate_proj zeroed, GPT-2-scale writer ──────────────────────
    ad = model_on.layers[0].audio_adapter
    assert torch.count_nonzero(ad.gate_proj.weight) == 0, "gate_proj must start zeroed"
    assert torch.count_nonzero(ad.up_proj.weight)   > 0,  "up_proj keeps default init"
    assert torch.count_nonzero(ad.down_proj.weight) > 0,  "down_proj keeps GPT-2 init"
    want_std = 0.02 / math.sqrt(cfg_on.n_layers)
    got_std  = ad.down_proj.weight.std().item()
    assert 0.5 * want_std < got_std < 2.0 * want_std, (
        f"down_proj std {got_std:.5f} should sit near GPT-2 scale {want_std:.5f}"
    )
    print("[OK] init: gate_proj=0, up_proj default, down_proj at GPT-2 scale")

    # ── Test: identity at init (gate_proj == 0 → exact no-op) ─────────────────
    with torch.no_grad():
        logits_on,  _ = model_on(inputs, audio_mask=mask)
        logits_off, _ = model_off(inputs, audio_mask=mask)
    assert torch.allclose(logits_on, logits_off, atol=1e-6), (
        f"adapter branch must be a no-op at init; max diff "
        f"{(logits_on - logits_off).abs().max().item():.2e}"
    )
    print("[OK] identity at init: r=16 matches r=0")

    # ── Test: mask correctness at a single block ───────────────────────────────
    # Whole-model logits would let causal attention spread an audio-position
    # perturbation to later positions.  A single block adds the masked branch
    # AFTER attention/MLP, so it perturbs ONLY masked positions.
    block = LlamaBlock(cfg_on, has_audio_adapter=True)
    block.eval()
    head_dim = cfg_on.d_model // cfg_on.n_heads
    cos, sin = _precompute_rope_cos_sin(head_dim, S, cfg_on.rope_theta)
    x = torch.randn(B, S, D)
    with torch.no_grad():
        ref = block(x, cos, sin, audio_mask=None)          # adapter disabled
        block.audio_adapter.gate_proj.weight.normal_(mean=0.0, std=0.05)
        out = block(x, cos, sin, mask)                     # adapter live at masked positions
    on = mask[..., 0].bool()
    assert not torch.allclose(out[on], ref[on]), "masked positions must change when live"
    assert torch.allclose(out[~on], ref[~on], atol=1e-6), "unmasked positions must be untouched"
    print("[OK] mask correctness: only masked positions change when the branch is live")

    # ── Test: the mask is honoured verbatim and shape-checked ──────────────────
    live = Llama(cfg_on)
    live.eval()
    _lift_gates(live)
    with torch.no_grad():
        by_offset, _ = live(inputs, audio_mask=mask)
        by_other,  _ = live(inputs, audio_mask=_span_mask(0, [4, 2]))
        by_none,   _ = live(inputs)
        by_zeros,  _ = live(inputs, audio_mask=torch.zeros(B, S, 1))
    assert not torch.allclose(by_offset, by_other, atol=1e-6), "mask position must matter"
    # An all-zero mask (the probe suite's text-only control) == no adapters at all.
    assert torch.allclose(by_none, by_zeros, atol=1e-6), "all-zero mask must equal mask=None"
    try:
        live(inputs, audio_mask=mask.squeeze(-1))
    except ValueError:
        pass
    else:
        raise AssertionError("a mis-shaped audio_mask must raise ValueError")
    print("[OK] audio_mask: position honoured, all-zero == None, shape-checked")

    # ── Test: gradient flow — gate_proj at step 0, everything from step 1 ──────
    # silu(0) = 0 makes the branch a no-op while silu'(0) = 0.5 keeps gate_proj's
    # gradient alive.  Run under gradient checkpointing, adapter-only regime:
    # pretrained params frozen, inputs require no grad.
    gc_model = Llama(cfg_on)
    gc_model.train()
    gc_model.enable_gradient_checkpointing()
    for name, p in gc_model.named_parameters():
        p.requires_grad_("audio_adapter" in name)
    labels = torch.full((B, S), -100, dtype=torch.long)
    labels[:, 7:] = torch.randint(0, cfg_on.vocab_size, (B, S - 7))

    def _adapter_grads() -> dict[str, float]:
        """One backward pass; return per-parameter |grad| sums for the adapters."""
        gc_model.zero_grad(set_to_none=True)
        _, loss = gc_model(torch.randn(B, S, D), labels, audio_mask=mask)
        loss.backward()
        return {
            n: (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for n, p in gc_model.named_parameters() if "audio_adapter" in n
        }

    g0 = _adapter_grads()
    assert all(v > 0 for n, v in g0.items() if "gate_proj" in n), (
        f"gate_proj must receive gradient at step 0 (silu'(0)=0.5): {g0}"
    )
    # Everything else is proportional to the (zero) gate output: exactly zero.
    assert all(v == 0.0 for n, v in g0.items() if "gate_proj" not in n), g0
    torch.optim.SGD([p for p in gc_model.parameters() if p.requires_grad], lr=1.0).step()
    g1 = _adapter_grads()
    assert all(v > 0 for v in g1.values()), f"every adapter param must train from step 1: {g1}"
    for name, p in gc_model.named_parameters():
        if "audio_adapter" not in name:
            assert p.grad is None, f"frozen param {name} should have no grad"
    print("[OK] gradients (checkpointed): gate_proj at step 0, all adapter params from step 1")

    # ── Test: helper methods ───────────────────────────────────────────────────
    assert not any("gate" in n and "gate_proj" not in n
                   for n, _ in model_on.named_parameters()), "scalar gate must be gone"
    scales = model_on.audio_adapter_scales()
    assert set(scales) == {
        f"adapter_scale/layer_{i:02d}_{k}"
        for i in range(cfg_on.n_layers - 1) for k in ("writer", "input")
    }, scales
    assert all(v == 0.0 for k, v in scales.items() if k.endswith("_input")),  scales
    assert all(v  > 0.0 for k, v in scales.items() if k.endswith("_writer")), scales
    assert model_off.audio_adapter_scales() == {}, "r=0 model exposes no adapter scales"
    assert model_on.audio_adapter_parameters() and not model_off.audio_adapter_parameters()
    print("[OK] audio_adapter_scales / audio_adapter_parameters helpers")

    print("\nPASSED")
    sys.exit(0)
