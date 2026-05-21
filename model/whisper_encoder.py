"""Whisper small encoder, implemented from scratch.

Zero dependency on `openai-whisper` or `transformers`. All subcomponents are
contained in this single file: conv stem, multi-head self-attention, layer norm,
MLP, learned positional embeddings, and transformer blocks (Pre-LN).

Architecture (Whisper small encoder):
    mel (B, 80, 3000)
    → Conv1d stem (stride=2) → (B, 512, 1500)
    → + positional embeddings → (B, 1500, 512)
    → 12 × TransformerBlock (Pre-LN) → (B, 1500, 512)
    → LayerNorm → (B, 1500, 512)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Architecture constants (Whisper small) ────────────────────────────────────
N_MELS      = 80
N_AUDIO_CTX = 1500   # output tokens (20 ms per token)
N_STATE     = 768    # d_model
N_HEAD      = 12
N_LAYER     = 12

# ── Audio preprocessing constants ────────────────────────────────────────────
SAMPLE_RATE = 16_000
N_FFT       = 400    # 25 ms window
HOP_LENGTH  = 160    # 10 ms hop
N_SAMPLES   = 30 * SAMPLE_RATE  # 480 000 samples = 30 s


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mel_filters(n_mels: int = N_MELS) -> torch.Tensor:
    """Compute triangular HTK mel filterbank matrix.

    Args:
        n_mels: number of mel frequency bands

    Returns:
        (n_mels, N_FFT // 2 + 1) float32 tensor
    """
    fmin, fmax = 0.0, SAMPLE_RATE / 2.0

    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)  # (n_mels + 2,)

    freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, N_FFT // 2 + 1)  # (N_FFT//2+1,)

    lo  = hz_pts[:-2, None]   # (n_mels, 1)
    mid = hz_pts[1:-1, None]  # (n_mels, 1)
    hi  = hz_pts[2:,  None]   # (n_mels, 1)
    f   = freqs[None, :]      # (1, N_FFT//2+1)

    up   = np.where((lo <= f) & (f <= mid), (f - lo) / np.maximum(mid - lo, 1e-10), 0.0)
    down = np.where((mid < f) & (f <= hi), (hi - f) / np.maximum(hi - mid, 1e-10), 0.0)

    return torch.from_numpy((up + down).astype(np.float32))


def log_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Compute a log-mel spectrogram from raw audio at its natural length.

    Pure function: no trimming, no padding. The caller is responsible for
    ensuring the audio fits within the encoder's positional embedding range
    (≤ 30 s / N_SAMPLES samples). Samples that exceed this should be
    discarded at preprocessing time rather than silently truncated here,
    because trimming audio without trimming the transcript introduces label noise.

    Output time dimension: T = T_audio // HOP_LENGTH.
    Batch-level padding to a uniform length is done in the DataLoader
    collation function (not here).

    Args:
        audio:       (T_audio,) float32, values normalised to [-1, 1]
        sample_rate: must equal 16 000 Hz

    Returns:
        (N_MELS, T) float32 log-mel spectrogram, T = T_audio // HOP_LENGTH
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected sample_rate={SAMPLE_RATE}, got {sample_rate}")

    # centre=True: reflect-pads by n_fft//2=200 on each side.
    # Frame count before drop: T_audio//160 + 1. Drop last boundary frame → T_audio//160.
    window     = torch.hann_window(N_FFT, device=audio.device)
    stft       = torch.stft(audio, N_FFT, HOP_LENGTH, window=window,
                            return_complex=True, center=True, pad_mode="reflect")
    magnitudes = stft[..., :-1].abs() ** 2  # (N_FFT//2+1, T)

    filters  = _mel_filters(N_MELS).to(audio.device)  # (80, N_FFT//2+1)
    mel_spec = filters @ magnitudes                     # (80, T)

    # Whisper's log-scale normalisation
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec  # (N_MELS, T), T = T_audio // HOP_LENGTH


# ── Subcomponents ─────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention used inside each encoder block.

    No causal mask — the encoder attends to all positions bidirectionally.
    Key projection has no bias, matching the OpenAI Whisper implementation.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Initialise Q, K, V, and output projections.

        Args:
            d_model:  model dimension (must be divisible by n_heads)
            n_heads:  number of attention heads
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)  # no bias in Whisper
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: (B, T, D)

        Returns:
            (B, T, D)
        """
        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # (B, H, T, D_h)

        # Uses FlashAttention-2 automatically when available
        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, T, D_h)
        out = out.transpose(1, 2).reshape(B, T, D)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: LayerNorm → Attention → residual,
    then LayerNorm → MLP → residual.

    Pre-LN (norm before sub-layer, not after) matches the OpenAI Whisper
    implementation and tends to train more stably without a warm-up schedule.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Initialise the block.

        Args:
            d_model: model dimension
            n_heads: attention heads
        """
        super().__init__()
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn    = MultiHeadSelfAttention(d_model, n_heads)
        self.mlp_ln  = nn.LayerNorm(d_model)
        self.mlp     = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block in-place via residual connections.

        Args:
            x: (B, T, D)

        Returns:
            (B, T, D)
        """
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x


# ── Main encoder ──────────────────────────────────────────────────────────────

class WhisperEncoder(nn.Module):
    """Whisper small encoder (encoder-only; decoder is discarded).

    Input:  (B, 80, T_mel) log-mel spectrogram, T_mel ≤ 3000 (= 30 s)
    Output: (B, T_mel//2, N_STATE) contextual audio embeddings

    Accepts variable-length mel inputs — the sequence length is determined
    dynamically from the input, not from the N_AUDIO_CTX constant. Callers
    are responsible for ensuring T_mel does not exceed the positional
    embedding size (N_AUDIO_CTX = 1500 encoder output tokens).

    Initialise with random weights for training from scratch, or call
    load_openai_weights() to use the pretrained OpenAI checkpoint.
    """

    def __init__(self) -> None:
        """Construct all layers using the Whisper small constants."""
        super().__init__()
        # Conv stem: (B, 80, 3000) → (B, 512, 3000) → (B, 512, 1500)
        self.conv1 = nn.Conv1d(N_MELS,  N_STATE, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(N_STATE, N_STATE, kernel_size=3, stride=2, padding=1)

        # Learned positional embeddings (OpenAI stores sinusoidal values here,
        # but they are stored as a trained parameter — we use nn.Embedding so
        # they are updated during joint training)
        self.pos_embedding = nn.Embedding(N_AUDIO_CTX, N_STATE)

        self.blocks  = nn.ModuleList(
            [TransformerBlock(N_STATE, N_HEAD) for _ in range(N_LAYER)]
        )
        self.ln_post = nn.LayerNorm(N_STATE)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode a batch of log-mel spectrograms.

        Args:
            mel: (B, N_MELS, T_mel) log-mel spectrogram, T_mel ≤ N_AUDIO_CTX * 2

        Returns:
            (B, T_mel//2, N_STATE) encoder hidden states
        """
        # Conv stem
        x = F.gelu(self.conv1(mel))   # (B, N_STATE, T_mel)
        x = F.gelu(self.conv2(x))     # (B, N_STATE, T_mel//2)

        # (B, D, T) → (B, T, D) and add positional embeddings
        x = x.permute(0, 2, 1)        # (B, T_mel//2, N_STATE)
        T = x.shape[1]                 # actual sequence length (varies per batch)
        pos = torch.arange(T, device=x.device)
        x = x + self.pos_embedding(pos)  # broadcast over batch

        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)  # (B, T_mel//2, N_STATE)

    def load_openai_weights(self, checkpoint_path: Path) -> None:
        """Load weights from an OpenAI Whisper small checkpoint (.pt file).

        Asserts on every parameter shape before copying. Any shape mismatch
        raises immediately with a descriptive message — silent broadcasts are
        silent bugs.

        Args:
            checkpoint_path: path to whisper_small.pt downloaded from OpenAI CDN
        """
        checkpoint_path = Path(checkpoint_path)
        # weights_only=False is required because the Whisper checkpoint contains a
        # non-tensor `dims` dict alongside the state dict.
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model_state_dict" in raw:
            src  = raw["model_state_dict"]
            dims = raw.get("dims", {})
        else:
            # Fallback: assume the file IS the state dict
            src  = raw
            dims = {}

        # Sanity-check the checkpoint dimensions match our constants
        if dims:
            checks = {
                "n_mels":       (dims.get("n_mels"),       N_MELS),
                "n_audio_ctx":  (dims.get("n_audio_ctx"),  N_AUDIO_CTX),
                "n_audio_state":(dims.get("n_audio_state"), N_STATE),
                "n_audio_head": (dims.get("n_audio_head"), N_HEAD),
                "n_audio_layer":(dims.get("n_audio_layer"), N_LAYER),
            }
            for name, (got, expected) in checks.items():
                if got is not None and got != expected:
                    raise ValueError(
                        f"Checkpoint {name}={got} does not match expected {expected}. "
                        "Did you download the wrong Whisper variant?"
                    )

        def _load(param: nn.Parameter, key: str) -> None:
            if key not in src:
                raise KeyError(f"Key missing from checkpoint: '{key}'")
            w = src[key]
            if w.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': "
                    f"checkpoint {tuple(w.shape)} vs model {tuple(param.shape)}"
                )
            with torch.no_grad():
                param.copy_(w)

        _load(self.conv1.weight, "encoder.conv1.weight")
        _load(self.conv1.bias,   "encoder.conv1.bias")
        _load(self.conv2.weight, "encoder.conv2.weight")
        _load(self.conv2.bias,   "encoder.conv2.bias")

        # OpenAI stores the positional embedding as a plain tensor (buffer),
        # not as nn.Embedding.weight — map it explicitly.
        _load(self.pos_embedding.weight, "encoder.positional_embedding")

        for i, block in enumerate(self.blocks):
            p = f"encoder.blocks.{i}"
            _load(block.attn_ln.weight,       f"{p}.attn_ln.weight")
            _load(block.attn_ln.bias,         f"{p}.attn_ln.bias")
            _load(block.attn.q_proj.weight,   f"{p}.attn.query.weight")
            _load(block.attn.q_proj.bias,     f"{p}.attn.query.bias")
            _load(block.attn.k_proj.weight,   f"{p}.attn.key.weight")
            # k_proj has no bias — intentionally omitted
            _load(block.attn.v_proj.weight,   f"{p}.attn.value.weight")
            _load(block.attn.v_proj.bias,     f"{p}.attn.value.bias")
            _load(block.attn.out_proj.weight, f"{p}.attn.out.weight")
            _load(block.attn.out_proj.bias,   f"{p}.attn.out.bias")
            _load(block.mlp_ln.weight,        f"{p}.mlp_ln.weight")
            _load(block.mlp_ln.bias,          f"{p}.mlp_ln.bias")
            _load(block.mlp[0].weight,        f"{p}.mlp.0.weight")
            _load(block.mlp[0].bias,          f"{p}.mlp.0.bias")
            _load(block.mlp[2].weight,        f"{p}.mlp.2.weight")
            _load(block.mlp[2].bias,          f"{p}.mlp.2.bias")

        _load(self.ln_post.weight, "encoder.ln_post.weight")
        _load(self.ln_post.bias,   "encoder.ln_post.bias")
