"""Adapter between the Whisper encoder and Llama.

Applies temporal mean pooling (factor=4) to reduce 1500 → 375 tokens, then
projects from d_model=768 to d_model=4096 via a 2-layer MLP with GELU.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dimensions that bridge encoder and LLM
_ENCODER_DIM = 768    # WhisperEncoder d_model
_HIDDEN_DIM  = 2048   # adapter MLP hidden size
_LLAMA_DIM   = 4096   # Llama d_model
_POOL_FACTOR = 4      # temporal mean-pool reduction (1500 → 375)


class AudioAdapter(nn.Module):
    """Temporal mean-pool + 2-layer MLP projecting encoder output to Llama's d_model.

    Input:  (B, 1500, 768)    — WhisperEncoder output
    Output: (B,  375, llama_dim) — ready to be passed to prepare_input()
    """

    def __init__(
        self,
        llama_dim: int = _LLAMA_DIM,
        pca_init_path: str | None = None,
    ) -> None:
        """Initialise with random weights, optionally loading a PCA-based init.

        Args:
            llama_dim:     output dimension; must match the Llama model's d_model.
                           Defaults to 4096 (full Llama 3.1 8B). Pass a smaller
                           value (e.g. 512) when using a stub config for testing.
            pca_init_path: path to a .pt file produced by
                           tools/compute_adapter_pca_init.py. When provided,
                           mlp[0].weight is replaced with the saved PCA basis and
                           mlp[0].bias is zeroed. mlp[2] is unaffected.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(_ENCODER_DIM, _HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(_HIDDEN_DIM, llama_dim),
        )
        # initialise output projection to near-zero so adapter starts as identity-ish
        nn.init.normal_(self.mlp[2].weight, mean=0.0, std=0.02 / math.sqrt(6))
        nn.init.zeros_(self.mlp[2].bias)

        if pca_init_path is not None:
            self._load_pca_init(pca_init_path)

    def _load_pca_init(self, path: str) -> None:
        """Replace mlp[0] weights with the PCA basis saved at *path*.

        The file must contain a dict with key 'weight' of shape
        (_HIDDEN_DIM, _ENCODER_DIM) = (2048, 768).
        """
        data   = torch.load(path, map_location="cpu", weights_only=True)
        weight = data["weight"]   # (2048, 768)

        expected = self.mlp[0].weight.shape
        if weight.shape != expected:
            raise ValueError(
                f"PCA weight shape {tuple(weight.shape)} does not match "
                f"first linear layer {tuple(expected)}"
            )

        with torch.no_grad():
            self.mlp[0].weight.copy_(weight)
            nn.init.zeros_(self.mlp[0].bias)

        if "explained_variance_ratio" in data:
            evr    = data["explained_variance_ratio"].float()
            cum_ev = evr.sum().item()
            print(
                f"PCA init loaded from '{path}'  "
                f"(cumulative explained variance: {cum_ev:.4f} = {cum_ev:.2%})"
            )

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Pool and project encoder hidden states.

        Accepts variable-length encoder output. If T is not an exact multiple
        of _POOL_FACTOR, the time dimension is zero-padded to the next multiple
        before pooling — at most 3 extra frames, averaged away immediately.

        Args:
            encoder_out: (B, T, 768) — WhisperEncoder output, T ≤ 1500

        Returns:
            (B, ceil(T/_POOL_FACTOR), llama_dim)
        """
        B, T, D = encoder_out.shape
        remainder = T % _POOL_FACTOR
        if remainder:
            # Pad time dimension so reshape is valid; padding is averaged away.
            encoder_out = F.pad(encoder_out, (0, 0, 0, _POOL_FACTOR - remainder))
            T = encoder_out.shape[1]
        x = encoder_out.reshape(B, T // _POOL_FACTOR, _POOL_FACTOR, D).mean(dim=2)
        return self.mlp(x)
