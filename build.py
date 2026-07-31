"""Shared model construction for speech-llm.

Used by training.py and tools/run_wer.py so both see identical initialisation logic.
"""

from __future__ import annotations

import json

import torch

from model.adapter import AudioAdapter
from model.llama import Llama, LlamaConfig
from model.whisper_encoder import WhisperEncoder
from utils.checkpoint import load_weights
from utils.config import Config


def build_models(
    cfg: Config,
    device: torch.device,
    *,
    train: bool = True,
    apply_init_from: bool = True,
) -> tuple[WhisperEncoder, AudioAdapter, Llama, list[str]]:
    """Instantiate and initialise all three model components.

    Args:
        cfg:              full run Config
        device:           target device
        train:            .train() all modules if True, .eval() otherwise
        apply_init_from:  apply cfg.model.init_from warm-start weights when set;
                          pass False in run_wer.py which overlays per-checkpoint
                          weights via load_weights directly after this call

    Load order:
      1. Architecture (stub or full)
      2. Pretrained weights (whisper + llama) if not stub
      3. Warm-start overlay from cfg.model.init_from if set and apply_init_from=True

    Returns:
        (encoder, adapter, llama, init_from_loaded) where init_from_loaded is the
        list of module names actually overlaid from cfg.model.init_from (empty
        when init_from is unset or apply_init_from=False).  The training loop
        seeds its checkpoint dirty set from this: encoder/llama warm-started from
        a delta diverge from pretrained and must be saved thereafter.
    """
    with (cfg.data.tokenizer / "pruned_config.json").open() as f:
        vocab_size = json.load(f)["vocab_size"]

    if cfg.model.stub:
        llama_cfg = LlamaConfig(
            **cfg.model.stub_dims,
            vocab_size=vocab_size,
            audio_adapter_r=cfg.model.audio_adapter_r,
        )
        llama_dim = cfg.model.stub_dims["d_model"]
        print("STUB mode: tiny randomly-initialised model (no pretrained weights).")
    else:
        llama_cfg = LlamaConfig(
            vocab_size=vocab_size,
            audio_adapter_r=cfg.model.audio_adapter_r,
        )
        llama_dim = 4096

    encoder = WhisperEncoder()
    adapter = AudioAdapter(
        llama_dim=llama_dim,
        pca_init_path=cfg.model.adapter_pca_init,
    )
    llama = Llama(llama_cfg)

    if not cfg.model.stub:
        if cfg.model.whisper_ckpt is None or cfg.model.llama_ckpt is None:
            raise ValueError(
                "model.whisper_ckpt and model.llama_ckpt are required when model.stub is false."
            )
        print("Loading Whisper encoder weights …")
        encoder.load_openai_weights(cfg.model.whisper_ckpt)

        vocab_map_path = cfg.data.tokenizer / "vocab_map.json"
        with vocab_map_path.open() as f:
            vocab_map = json.load(f)
        print("Loading Llama transformer weights …")
        llama.load_meta_weights(cfg.model.llama_ckpt, vocab_map=vocab_map)
    elif cfg.model.whisper_ckpt is not None:
        print("Loading Whisper encoder weights (stub mode) …")
        encoder.load_openai_weights(cfg.model.whisper_ckpt)

    init_from_loaded: list[str] = []
    if apply_init_from and cfg.model.init_from is not None:
        print(f"Warm-starting from {cfg.model.init_from} …")
        init_from_loaded = load_weights(
            cfg.model.init_from, encoder=encoder, adapter=adapter, llama=llama
        )
        print(f"  Warm-start complete (loaded: {init_from_loaded}).")

    encoder = encoder.to(device)
    adapter = adapter.to(device)
    llama   = llama.to(device)

    if cfg.model.gradient_checkpointing:
        if cfg.model.stub:
            print("[warn] model.gradient_checkpointing ignored in stub mode")
        elif train:
            llama.enable_gradient_checkpointing()
            print("[info] gradient checkpointing enabled on Llama")
        # eval mode: gradient checkpointing irrelevant, skip silently

    if train:
        encoder.train()
        adapter.train()
        llama.train()
    else:
        encoder.eval()
        adapter.eval()
        llama.eval()

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_ada = sum(p.numel() for p in adapter.parameters())
    # Report the gated audio adapters separately and EXCLUDE them from the llama
    # count so the pretrained figure stays recognisable (~8B).
    n_aa  = sum(p.numel() for name, p in llama.named_parameters() if "audio_adapter" in name)
    n_llm = sum(p.numel() for name, p in llama.named_parameters() if "audio_adapter" not in name)
    msg = (
        f"Parameters — encoder: {n_enc / 1e6:.1f}M  "
        f"adapter: {n_ada / 1e6:.1f}M  "
        f"llama: {n_llm / 1e6:.0f}M  "
    )
    if n_aa > 0:
        msg += f"audio_adapters: {n_aa / 1e6:.2f}M  "
    msg += f"total: {(n_enc + n_ada + n_llm + n_aa) / 1e9:.2f}B"
    print(msg)
    return encoder, adapter, llama, init_from_loaded
