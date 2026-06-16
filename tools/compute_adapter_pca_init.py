"""Compute PCA-based weight initialisation for AudioAdapter's first linear layer.

Streams audio shards through the pretrained Whisper encoder (frozen), applies the
same temporal mean-pool as AudioAdapter (factor 4), collects pooled frame vectors,
runs truncated SVD, and saves the top principal components as a drop-in weight
initialisation for AudioAdapter.mlp[0].weight.

The Whisper encoder outputs 768-dim vectors. SVD on the (N, 768) data matrix yields
a (768, 768) Vt whose rows are the principal components. To fill all 2048 rows of
the adapter weight matrix, 2048 random coefficient vectors are sampled on the unit
sphere in R^768 and each is projected through Vt, giving rows that are unit-norm
random linear combinations of the principal components.

Usage:
    python scripts/compute_adapter_pca_init.py \\
        --shards_file  data/dev_train_shards.txt \\
        --whisper_ckpt weights/whisper_small.pt \\
        --output       data/adapter_pca_init.pt
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from tqdm import tqdm

# Ensure repo root is importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.whisper_encoder import WhisperEncoder, N_STATE

_POOL_FACTOR = 4
_ENCODER_DIM = N_STATE   # 768 for Whisper small
_HIDDEN_DIM  = 2048      # AudioAdapter first linear output dim
_MAX_ROWS    = 500_000   # subsample cap before SVD to keep runtime manageable
_MEL_MAX_T   = 3000      # 30 s = encoder positional embedding boundary


def _temporal_pool(x: torch.Tensor) -> torch.Tensor:
    """Factor-4 temporal mean-pool: (B, T, D) → (B, ceil(T/4), D).

    Mirrors AudioAdapter.forward exactly. Any change to the adapter pooling
    must be reflected here to keep the PCA basis and the adapter in sync.
    """
    B, T, D = x.shape
    rem = T % _POOL_FACTOR
    if rem:
        x = F.pad(x, (0, 0, 0, _POOL_FACTOR - rem))
        T = x.shape[1]
    return x.reshape(B, T // _POOL_FACTOR, _POOL_FACTOR, D).mean(dim=2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute PCA init for AudioAdapter's first linear layer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--shards_file", type=Path, required=True,
        help="Text file listing shard paths, one per line.",
    )
    p.add_argument(
        "--whisper_ckpt", type=Path, required=True,
        help="Path to whisper_small.pt pretrained checkpoint.",
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/adapter_pca_init.pt"),
        help="Destination .pt file for the PCA init dict.",
    )
    p.add_argument(
        "--batch_size", type=int, default=16,
        help="Mel spectrograms per encoder forward pass. Tune to fit VRAM.",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:  # noqa: C901
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── Load pretrained Whisper encoder (frozen, no grad) ─────────────────────
    encoder = WhisperEncoder().to(device)
    encoder.load_openai_weights(args.whisper_ckpt)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Whisper encoder loaded from {args.whisper_ckpt}  "
          f"(output dim: {_ENCODER_DIM}, frozen)")

    # ── Shard list ────────────────────────────────────────────────────────────
    raw_lines = args.shards_file.read_text().splitlines()
    shards    = [ln.strip() for ln in raw_lines if ln.strip()]
    if not shards:
        raise FileNotFoundError(f"No shard paths found in {args.shards_file}")
    print(f"Streaming from {len(shards)} shard(s) in {args.shards_file}")

    # ── Stream, encode, pool, accumulate ──────────────────────────────────────
    def _load_mel(sample: dict) -> np.ndarray:
        return np.load(io.BytesIO(sample["mel.npy"])).astype(np.float32)  # (80, T)

    dataset = wds.WebDataset(shards, shardshuffle=False).map(_load_mel)

    chunks:  list[torch.Tensor] = []
    total_rows = 0
    pending:  list[np.ndarray] = []

    def _flush(batch: list[np.ndarray]) -> None:
        nonlocal total_rows

        T_list = [m.shape[1] for m in batch]
        T_max  = math.ceil(max(T_list) / 8) * 8   # multiple-of-8 padding, same as DataLoader
        B      = len(batch)

        mel_t = torch.zeros(B, 80, T_max, dtype=torch.float32, device=device)
        for i, m in enumerate(batch):
            mel_t[i, :, : m.shape[1]] = torch.from_numpy(m).to(device)

        # Real adapter token count per sample (CLAUDE.md Decision 007):
        #   encoder conv stride-2 → T_enc = T_mel // 2
        #   adapter mean-pool-4   → T_pool = ceil(T_enc / 4) = (T_enc + 3) // 4
        pool_lens = [(T // 2 + 3) // 4 for T in T_list]

        with torch.no_grad():
            enc = encoder(mel_t)        # (B, T_max//2, 768)
            pol = _temporal_pool(enc)   # (B, ceil(T_max/8), 768)

        # Collect only real (non-padding) frames for each sample.
        for i, n_real in enumerate(pool_lens):
            frames = pol[i, :n_real, :].cpu().float()   # (n_real, 768)
            chunks.append(frames)
            total_rows += n_real

    pbar = tqdm(dataset, desc="Encoding shards")
    for mel_arr in pbar:
        if mel_arr.shape[1] > _MEL_MAX_T:
            # Skip utterances longer than 30 s; they exceed the positional embedding range.
            continue
        pending.append(mel_arr)
        if len(pending) == args.batch_size:
            _flush(pending)
            pending.clear()
            pbar.set_postfix({"pooled_frames": f"{total_rows:,}"})

    if pending:
        _flush(pending)
        pending.clear()

    print(f"\nCollected {total_rows:,} pooled feature vectors  (dim={_ENCODER_DIM})")

    # ── Build full data matrix ────────────────────────────────────────────────
    X = torch.cat(chunks, dim=0)   # (N, 768), CPU float32
    del chunks

    if total_rows > _MAX_ROWS:
        rng = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(total_rows, generator=rng)[:_MAX_ROWS]
        X   = X[idx].contiguous()
        print(f"Subsampled to {_MAX_ROWS:,} rows before SVD.")

    # Centre the data (standard PCA pre-processing).
    X -= X.mean(dim=0, keepdim=True)

    # ── Truncated SVD ─────────────────────────────────────────────────────────
    # X: (N, D) with D=768.  full_matrices=False gives:
    #   U  : (N, D)   — left singular vectors
    #   S  : (D,)     — singular values in descending order
    #   Vt : (D, D)   — rows are principal components (right singular vectors)
    print(f"Running SVD on ({X.shape[0]:,} × {X.shape[1]}) matrix …")
    _, S, Vt = torch.linalg.svd(X, full_matrices=False)
    del X

    # Explained variance: proportional to squared singular values.
    variance  = S.float() ** 2
    total_var = variance.sum()
    evr       = variance / total_var   # (D,) per-component ratio

    n_pcs_available = Vt.shape[0]   # = _ENCODER_DIM = 768
    cum_ev          = evr.sum().item()
    print(
        f"\nPCA statistics:"
        f"\n  principal components available  : {n_pcs_available}"
        f"\n  adapter hidden dim (rows)        : {_HIDDEN_DIM}"
        f"\n  cumulative explained variance    : {cum_ev:.6f}  ({cum_ev:.2%})"
    )

    # ── Build weight matrix  (2048, 768) ─────────────────────────────────────
    # Each row is a random linear combination of all 768 PCs, with the
    # combination coefficients normalised to unit L2 norm.  Because Vt is
    # orthogonal, multiplying by Vt preserves norms: ||w_i||_2 = ||c_i||_2 = 1.
    rng = torch.Generator().manual_seed(args.seed)
    C      = torch.randn(_HIDDEN_DIM, n_pcs_available, dtype=torch.float32, generator=rng)
    C      = C / C.norm(dim=1, keepdim=True)   # (2048, 768) unit-norm rows
    weight = C @ Vt.float()                     # (2048, 768)

    # ── Save ──────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weight":                   weight,          # (2048, 768) float32
            "explained_variance_ratio": evr,             # (768,)  float32
        },
        args.output,
    )
    print(f"\nSaved → {args.output}  (weight shape: {tuple(weight.shape)})")


if __name__ == "__main__":
    main()
