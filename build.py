"""Shared model construction for speech-llm.

Used by training.py and tools/run_wer.py so both see identical initialisation logic.
"""

from __future__ import annotations

import json

import torch

from data import load_pruned_config
from model.adapter import build_bridge_adapter, BridgeAdapter
from model.llama import Llama, LlamaConfig
from model.whisper_encoder import WhisperEncoder
from utils.checkpoint import load_weights
from utils.config import Config


def _report_special_rows(llama: Llama, vocab_map: dict, tokenizer_dir) -> None:
    """Print which chat-special embedding rows came from the pretrained table.

    Loud by design: the entire point of forcing the four chat specials into the
    vocabulary is that <|eot_id|> arrives carrying the Instruct checkpoint's
    termination prior.  A randomly-initialised eot row would still train, still
    terminate eventually, and quietly waste the transfer — so state plainly that
    each row was taken from the checkpoint, and fail if one was not.

    Args:
        llama:         the model whose embedding was just loaded
        vocab_map:     {str(old_id): new_id} used for the pruned embedding init
        tokenizer_dir: pruned tokenizer directory (for the error message)
    """
    pruned = load_pruned_config(tokenizer_dir)
    ids    = {k: v for k, v in pruned.chat_special_ids.items() if v is not None}
    if not ids:
        print("[build] vocabulary defines no chat specials (flat-convention vocab)")
        return

    reverse = {new: int(old) for old, new in vocab_map.items()}
    rows    = llama.embed_tokens.weight
    for key, new_id in sorted(ids.items(), key=lambda kv: kv[1]):
        if new_id not in reverse:
            raise ValueError(
                f"{key}: pruned id {new_id} is absent from vocab_map — the "
                f"vocabulary in {tokenizer_dir} is inconsistent with itself"
            )
        row = rows[new_id]
        # A pretrained embedding row is never all-zero; a row the loader never
        # touched would still hold the small-std fresh init.  Report the norm so
        # the log carries the evidence rather than just a claim.
        print(
            f"[build] {key:<16} pruned id {new_id:>6} ← original "
            f"{reverse[new_id]:>6}  |row| = {row.norm().item():.4f}"
        )


def build_models(
    cfg: Config,
    device: torch.device,
    *,
    train: bool = True,
    apply_init_from: bool = True,
) -> tuple[WhisperEncoder, BridgeAdapter, Llama, list[str]]:
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
    vocab_size = load_pruned_config(cfg.data.tokenizer).vocab_size

    if cfg.model.stub:
        llama_cfg = LlamaConfig(
            **cfg.model.stub_dims,
            vocab_size=vocab_size,
            audio_adapter_r=cfg.model.audio_adapter_r,
            audio_adapter_type=cfg.model.audio_adapter_type,
            audio_adapter_zero_writer_layers=cfg.model.audio_adapter_zero_writer_layers,
        )
        llama_dim = cfg.model.stub_dims["d_model"]
        print("STUB mode: tiny randomly-initialised model (no pretrained weights).")
    else:
        llama_cfg = LlamaConfig(
            vocab_size=vocab_size,
            audio_adapter_r=cfg.model.audio_adapter_r,
            audio_adapter_type=cfg.model.audio_adapter_type,
            audio_adapter_zero_writer_layers=cfg.model.audio_adapter_zero_writer_layers,
        )
        llama_dim = 4096

    encoder = WhisperEncoder()
    adapter = build_bridge_adapter(
        cfg.model.bridge_type,
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
        _report_special_rows(llama, vocab_map, cfg.data.tokenizer)
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

    # No stub exemption here: the encoder is the real 88M Whisper module even in
    # stub mode.  The flag is self-gating — the encoder's forward skips
    # recomputation while the module is frozen — so it costs nothing to leave on
    # in adapter-only stages and pays off the moment the encoder unfreezes.
    if cfg.model.encoder_gradient_checkpointing and train:
        encoder.enable_gradient_checkpointing()
        print("[info] gradient checkpointing enabled on Whisper encoder")

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
