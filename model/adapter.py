"""Bridge between the Whisper encoder and Llama.

Applies temporal mean pooling (factor=4) to reduce 1500 → 375 tokens, then
projects from d_model=768 to d_model=4096 with a SwiGLU
(:class:`AudioSwiGLUBridge`).  The output projection starts NEAR zero so the
audio embeddings entering Llama are tiny at step 0.

The bridge also carries the two learned audio delimiter vectors ``audio_bos`` /
``audio_eos`` that the chat sequence splices around the audio
(model/sequence.py).  They ride in the bridge's parameter group and state_dict.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dimensions that bridge encoder and LLM
_ENCODER_DIM = 768    # WhisperEncoder d_model
_HIDDEN_DIM  = 2048   # SwiGLU hidden size
_LLAMA_DIM   = 4096   # Llama d_model
_POOL_FACTOR = 4      # temporal mean-pool reduction (1500 → 375)


def _pool_time(encoder_out: torch.Tensor) -> torch.Tensor:
    """Mean-pool the time axis by _POOL_FACTOR, padding to a whole multiple first.

    If T is not an exact multiple of _POOL_FACTOR the time dimension is zero-padded to the next multiple — at
    most 3 extra frames, averaged away immediately.

    Args:
        encoder_out: (B, T, D)

    Returns:
        (B, ceil(T/_POOL_FACTOR), D)
    """
    B, T, D = encoder_out.shape
    remainder = T % _POOL_FACTOR
    if remainder:
        encoder_out = F.pad(encoder_out, (0, 0, 0, _POOL_FACTOR - remainder))
        T = encoder_out.shape[1]
    return encoder_out.reshape(B, T // _POOL_FACTOR, _POOL_FACTOR, D).mean(dim=2)


# Learned audio delimiter vectors (chat convention, model/sequence.py).  They are
# input-only embeddings — deliberately NOT vocabulary tokens, so they own no
# logit row and can never be generated.
_MARKER_INIT_STD = 0.02


def _init_audio_markers(module: nn.Module, llama_dim: int) -> None:
    """Give *module* the two learned AUDIO_BOS / AUDIO_EOS marker vectors.

    The markers live on the bridge (not on Llama) for two reasons: they are part
    of the audio-injection interface, and it puts them in the existing "adapter"
    parameter group and the always-saved adapter state_dict for free — no new
    module name in stages._MODULE_ORDER, no checkpoint-delta change.

    Init is an ordinary normal(0, 0.02) — the near-zero rule that governs the
    bridge's output projection does NOT apply here.  That rule exists because the
    bridge output is the ENTIRE content of the audio positions, so an exactly-zero
    write leaves a dead residual stream through a bias-free Llama.  The markers
    are two positions among live scaffold embeddings, so they are never the whole
    stream; normal init just matches the scale of the token embeddings beside them.

    Args:
        module:    the bridge to attach the markers to
        llama_dim: Llama d_model — the marker vectors' dimension
    """
    module.audio_bos = nn.Parameter(torch.empty(llama_dim))
    module.audio_eos = nn.Parameter(torch.empty(llama_dim))
    nn.init.normal_(module.audio_bos, mean=0.0, std=_MARKER_INIT_STD)
    nn.init.normal_(module.audio_eos, mean=0.0, std=_MARKER_INIT_STD)


class AudioSwiGLUBridge(nn.Module):
    """Temporal mean-pool + SwiGLU projecting encoder output to Llama's d_model.

        down_proj(silu(gate_proj(x)) ⊙ up_proj(x))

    Naming follows Llama's SwiGLUMLP: ``gate_proj``/``up_proj`` are the two
    768→2048 input projections and ``down_proj`` (2048→llama_dim) writes the
    audio embeddings that Llama consumes.

    ``down_proj`` gets a NEAR-zero init (std = 0.02/sqrt(6), zero bias), so the audio embeddings start
    tiny and the bridge is a near-no-op at step 0.  Its gradient is proportional
    to the non-zero SwiGLU hidden state, so it trains from the first step; the
    input projections join once it has moved.  Do NOT zero an input projection:
    that would make every gradient in the bridge identically zero.

    Why near-zero and not EXACTLY zero (unlike the in-layer adapters, whose
    writers are zeroed): the bridge output is the entire content of the audio
    positions, and Llama is bias-free, so exactly-zero embeddings stay exactly
    zero through every layer — attention(0)=0 and mlp(0)=0.  Each RMSNorm on
    that dead stream then contributes a backward gain of 1/sqrt(eps) ≈ 316×
    (rsqrt(0 + eps)), and across 32 layers × 2 norms the gradient reaching this
    module hits ~1e7 — far past fp16's 65504, so autocast produces inf, the
    GradScaler skips the step, the weights never move, and the degenerate state
    persists.  Measured: exact zero → dL/d(audio emb) absmax 1.5e7 (overflows);
    near-zero → 5.3e-3 (healthy).

    Input:  (B, 1500, 768)      — WhisperEncoder output
    Output: (B,  375, llama_dim) — ready to be passed to prepare_input()
    """

    def __init__(self, llama_dim: int = _LLAMA_DIM) -> None:
        """Initialise the SwiGLU with a near-zero output projection.

        Args:
            llama_dim: output dimension; must match the Llama model's d_model
                       (4096 for Llama 3.1 8B, smaller for a stub config)
        """
        super().__init__()
        self.gate_proj = nn.Linear(_ENCODER_DIM, _HIDDEN_DIM)   # 768 → 2048 (SwiGLU gate)
        self.up_proj   = nn.Linear(_ENCODER_DIM, _HIDDEN_DIM)   # 768 → 2048 (SwiGLU value)
        self.down_proj = nn.Linear(_HIDDEN_DIM,  llama_dim)     # 2048 → llama_dim (writes out)

        # Near-zero output projection → audio embeddings start tiny but NOT zero.
        # Exactly zero is a degenerate fixed point of the frozen stack; see the
        # class docstring before changing this to zeros_.
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02 / math.sqrt(6))
        nn.init.zeros_(self.down_proj.bias)

        # Audio delimiters; see _init_audio_markers for why plain normal init is safe.
        _init_audio_markers(self, llama_dim)

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Pool and project encoder hidden states.

        Args:
            encoder_out: (B, T, 768) — WhisperEncoder output, T ≤ 1500

        Returns:
            (B, ceil(T/_POOL_FACTOR), llama_dim)
        """
        x = _pool_time(encoder_out)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    torch.manual_seed(0)

    B, T, LLAMA_DIM = 2, 10, 64          # T not a multiple of 4 → exercises padding
    enc_out = torch.randn(B, T, _ENCODER_DIM)

    sw_bridge = AudioSwiGLUBridge(llama_dim=LLAMA_DIM)

    # ── Test: shapes and pooling ───────────────────────────────────────────────
    want_T = -(-T // _POOL_FACTOR)        # ceil
    out = sw_bridge(enc_out)
    assert out.shape == (B, want_T, LLAMA_DIM), tuple(out.shape)
    # Exact-multiple input pools without padding.
    assert sw_bridge(torch.randn(B, 8, _ENCODER_DIM)).shape == (B, 2, LLAMA_DIM)
    print(f"[OK] bridge pools {T} → {want_T} frames and projects to {LLAMA_DIM}")

    # ── Test: writes near-zero — but NOT zero — at init ───────────────────────
    # Exactly-zero audio embeddings are a degenerate fixed point: bias-free Llama
    # propagates them unchanged through every layer and each RMSNorm contributes
    # a 1/sqrt(eps) ≈ 316× backward gain, overflowing fp16.  See the class
    # docstring; this assertion is what stops a well-meaning zeros_ from
    # reintroducing it.
    with torch.no_grad():
        sw_out = sw_bridge(enc_out)
    assert torch.count_nonzero(sw_out) > 0, "down_proj must NOT be exactly zero"
    sw_rms = sw_out.pow(2).mean().sqrt()
    assert 1e-4 < sw_rms < 0.1, f"output should start near (not at) zero, got RMS {sw_rms:.3e}"
    assert torch.count_nonzero(sw_bridge.gate_proj.weight) > 0, "gate_proj keeps default init"
    assert torch.count_nonzero(sw_bridge.up_proj.weight)   > 0, "up_proj keeps default init"
    print(f"[OK] bridge: near-zero output at init (RMS {sw_rms:.2e}), no dead audio stream")

    # ── Test: gradients — every parameter trains from step 0 ──────────────────
    def _grads(module: nn.Module) -> dict[str, float]:
        """One backward pass; return per-parameter |grad| sums.

        The loss must have a non-zero gradient AT a zero output — an MSE against
        a zero target would not (d/dout of mean(out²) is 0 when out is 0), which
        would make this test measure the loss, not the module.  Real training is
        cross-entropy downstream of Llama, where the upstream gradient is
        non-zero regardless.
        """
        module.zero_grad(set_to_none=True)
        out    = module(torch.randn(B, T, _ENCODER_DIM))
        target = torch.randn_like(out)
        (out - target).pow(2).mean().backward()
        # grad is None for the marker vectors: bridge.forward never touches them
        # (prepare_input splices them into the sequence).  Report them as 0.0
        # rather than crashing; the caller filters them out.
        return {
            n: (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for n, p in module.named_parameters()
        }

    # The markers are spliced by prepare_input, not by bridge.forward, so they
    # legitimately get no gradient from this projection-only backward — they are
    # exercised in the marker test below instead.
    g0 = {n: v for n, v in _grads(sw_bridge).items() if "audio_" not in n}
    assert all(v > 0 for v in g0.values()), f"every bridge param must train at step 0: {g0}"
    print("[OK] bridge: every projection parameter receives gradient at step 0")

    # ── Test: audio marker vectors ────────────────────────────────────────────
    # They must exist, be non-zero (they sit among live scaffold
    # embeddings — the near-zero rule does not apply), train, and be captured by
    # state_dict() so save_adapter_checkpoint carries them.
    for marker in ("audio_bos", "audio_eos"):
        vec = getattr(sw_bridge, marker)
        assert isinstance(vec, nn.Parameter),      f"{marker} must be a Parameter"
        assert vec.shape == (LLAMA_DIM,),          f"{marker}: {tuple(vec.shape)}"
        assert torch.count_nonzero(vec) > 0,       f"{marker} must not be zero-init"
        assert marker in sw_bridge.state_dict(),   f"{marker} missing from state_dict"
    # Not the same vector: two distinct delimiters, independently initialised.
    assert not torch.equal(sw_bridge.audio_bos, sw_bridge.audio_eos)

    # Gradient reaches them when they are used (as prepare_input uses them).
    sw_bridge.zero_grad(set_to_none=True)
    (sw_bridge.audio_bos.sum() * 2 + sw_bridge.audio_eos.pow(2).sum()).backward()
    assert sw_bridge.audio_bos.grad is not None and sw_bridge.audio_bos.grad.abs().sum() > 0
    assert sw_bridge.audio_eos.grad is not None and sw_bridge.audio_eos.grad.abs().sum() > 0
    print("[OK] audio_bos/audio_eos exist, train, and are in state_dict()")

    print("\nPASSED")
    sys.exit(0)
