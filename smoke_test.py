"""End-to-end forward + backward smoke test.

Uses a tiny stub Llama config so the full pipeline fits on an RTX 3060 (12 GB).
All weights are randomly initialised — no checkpoints needed.

Checks:
  1. Instantiate WhisperEncoder, AudioAdapter, Llama (stub config)
  2. Print parameter counts per component and total
  3. Build a dummy batch with variable-length audio, instructions, transcripts
  4. Full forward pass: mel → encoder → adapter → prepare_input → llama → loss
  5. loss.backward()
  6. Assert non-zero gradients for ≥1 parameter in each component
  7. Print peak VRAM via torch.cuda.max_memory_allocated()
  8. Assert peak VRAM < 10 GB

Run with:
    source ~/.venvs/aidev/bin/activate
    python smoke_test.py
"""

from __future__ import annotations

import torch

from model.adapter import AudioAdapter, prepare_input
from model.llama import Llama, LlamaConfig
from model.whisper_encoder import WhisperEncoder


def main() -> None:
    """Run the smoke test; raises AssertionError on any failed check."""
    print("=" * 60)
    print("speech-llm smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ── Stub config (tiny model that fits on the RTX 3060) ────────────────
    llama_cfg = LlamaConfig(
        n_layers=2, d_model=512, n_heads=8, n_kv_heads=2,
        intermediate_size=1024, vocab_size=512,
    )

    # ── Instantiate ───────────────────────────────────────────────────────
    encoder = WhisperEncoder().to(device)
    adapter = AudioAdapter(llama_dim=llama_cfg.d_model).to(device)
    llama   = Llama(llama_cfg).to(device)

    # ── Parameter counts (PyTorch deduplicates tied weights) ──────────────
    enc_params = sum(p.numel() for p in encoder.parameters())
    ada_params = sum(p.numel() for p in adapter.parameters())
    llm_params = sum(p.numel() for p in llama.parameters())
    total      = enc_params + ada_params + llm_params

    print(f"\nParameter counts:")
    print(f"  WhisperEncoder : {enc_params / 1e6:8.2f} M")
    print(f"  AudioAdapter   : {ada_params / 1e6:8.2f} M")
    print(f"  Llama (stub)   : {llm_params / 1e6:8.2f} M")
    print(f"  Total          : {total      / 1e6:8.2f} M")

    # ── Dummy batch ───────────────────────────────────────────────────────
    B         = 2
    SEP_ID    = 1
    VOCAB     = llama_cfg.vocab_size

    mel = torch.randn(B, 80, 3000, device=device)

    # Variable-length audio: sample 0 = full 30 s (375 tokens),
    #                        sample 1 = ~16.7 s (250 tokens)
    audio_lengths = torch.tensor([375, 250], device=device)

    # Variable-length instructions (padded to T_inst_max=20)
    T_inst_max   = 20
    inst_ids     = torch.randint(2, VOCAB, (B, T_inst_max), device=device)
    inst_lengths = torch.tensor([T_inst_max, 15], device=device)

    # Variable-length transcripts (padded to T_trans_max=30)
    T_trans_max   = 30
    trans_ids     = torch.randint(2, VOCAB, (B, T_trans_max), device=device)
    trans_lengths = torch.tensor([25, T_trans_max], device=device)

    print(f"\nDummy batch  (B={B}):")
    print(f"  mel shape       : {tuple(mel.shape)}")
    print(f"  audio_lengths   : {audio_lengths.tolist()}")
    print(f"  inst_lengths    : {inst_lengths.tolist()}")
    print(f"  trans_lengths   : {trans_lengths.tolist()}")

    # ── Forward pass ──────────────────────────────────────────────────────
    print("\nForward pass ...")

    enc_out     = encoder(mel)          # (B, 1500, 768)
    adapter_out = adapter(enc_out)      # (B, 375,  512)

    inputs, labels = prepare_input(
        adapter_out,
        audio_lengths,
        inst_ids,  inst_lengths,
        trans_ids, trans_lengths,
        llama.embed_tokens,
        SEP_ID,
    )

    logits, loss = llama(inputs, labels)

    assert loss is not None, "loss is None — check that labels were passed correctly"

    # L_max = max(375+1+20+1+25, 250+1+15+1+30) = max(422, 297) = 422
    L_max = max(
        int(audio_lengths[i]) + 1 + int(inst_lengths[i]) + 1 + int(trans_lengths[i])
        for i in range(B)
    )
    assert inputs.shape == (B, L_max, llama_cfg.d_model), \
        f"Unexpected inputs shape: {tuple(inputs.shape)}"
    assert logits.shape == (B, L_max, VOCAB), \
        f"Unexpected logits shape: {tuple(logits.shape)}"

    print(f"  encoder out  : {tuple(enc_out.shape)}")
    print(f"  adapter out  : {tuple(adapter_out.shape)}")
    print(f"  llama inputs : {tuple(inputs.shape)}   (L_max={L_max})")
    print(f"  logits       : {tuple(logits.shape)}")
    print(f"  loss         : {loss.item():.4f}")

    # ── Backward pass ─────────────────────────────────────────────────────
    print("\nBackward pass ...")
    loss.backward()

    components: dict[str, torch.nn.Module] = {
        "WhisperEncoder": encoder,
        "AudioAdapter":   adapter,
        "Llama":          llama,
    }
    for name, model in components.items():
        live = [p for p in model.parameters()
                if p.grad is not None and p.grad.abs().sum() > 0]
        assert live, f"{name}: no non-zero gradients — gradient flow is broken"
        print(f"  {name}: {len(live)} params with non-zero grad  ✓")

    # ── VRAM check ────────────────────────────────────────────────────────
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\nPeak VRAM : {peak_gb:.2f} GB")
        assert peak_gb < 10.0, \
            f"Peak VRAM {peak_gb:.2f} GB exceeds the 10 GB limit"
        print("VRAM check: OK (< 10 GB)")
    else:
        print("\nNo CUDA device — skipping VRAM check")

    print("\n" + "=" * 60)
    print("Smoke test  PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
