"""Model-equivalence regression gate for refactors.

Captures one deterministic forward pass of the stub model, then later proves
that the code still computes the same thing from the same weights.  Written
before the phase-2 cleanup so that deleting code paths can be checked against a
reference produced by the code as it was.

    python tools/equivalence_check.py            # capture (overwrites)
    python tools/equivalence_check.py --verify   # reload + compare

Capture:
  1. build the model from --config (default configs/stub_chat.yaml) under a
     fixed seed, on CPU, in fp32
  2. give every exactly-zero parameter tensor small random values — the
     zero-initialised gate_proj of each in-layer audio adapter would otherwise
     make the adapter branch an exact no-op, and the check would never exercise
     it
  3. save the three state_dicts as a checkpoint
  4. assemble a fixed batch from the first utterances of the stub diag shard
     (tarfile member order, half per instruction variant), run one
     teacher-forced forward pass, and a short greedy decode
  5. save batch, logits, loss and generated ids next to the checkpoint

Verify:
  1. build the model from the CURRENT code and load the checkpoint with
     strict=True — a renamed, added or dropped parameter fails here
  2. rebuild the batch and assert it equals the saved one (tokenizer and
     collation drift), then run the forward pass on the SAVED batch
  3. assert logits and loss match (atol 1e-5) and greedy ids match exactly

Everything lands under --out-dir (default out/equivalence/, gitignored).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from build import build_models                                     # noqa: E402
from data import INSTRUCTION_VARIANTS, PrunedTokenizer, read_eval_shard  # noqa: E402
from model.sequence import ChatTemplate, prepare_input             # noqa: E402
from utils.config import load_config                               # noqa: E402
from utils.generate import greedy_generate                         # noqa: E402

_ATOL = 1e-5
_N_SAMPLES = 4
_MAX_NEW_TOKENS = 8


def _seed_all(seed: int) -> None:
    """Seed every RNG that model construction touches."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fill_zero_tensors(modules: dict[str, torch.nn.Module], seed: int) -> list[str]:
    """Replace every all-zero parameter with small deterministic noise.

    Returns the names touched, so the capture log shows what was perturbed.
    """
    gen = torch.Generator().manual_seed(seed)
    touched: list[str] = []
    with torch.no_grad():
        for mod_name, mod in modules.items():
            for name, p in mod.named_parameters():
                if p.numel() and torch.count_nonzero(p) == 0:
                    p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
                    touched.append(f"{mod_name}.{name}")
    return touched


def _build_batch(shard: Path, tokenizer: PrunedTokenizer) -> dict[str, torch.Tensor]:
    """Collate the first _N_SAMPLES utterances of *shard* into one training batch.

    First half uses the unformatted instruction/label, second half formatted.
    Mel padding and audio_lengths follow data.build_dataloader's collation.
    """
    samples = read_eval_shard(shard)[:_N_SAMPLES]
    if len(samples) < _N_SAMPLES:
        raise ValueError(f"{shard}: need {_N_SAMPLES} samples, found {len(samples)}")

    B = len(samples)
    T_max = math.ceil(max(s.mel.shape[1] for s in samples) / 8) * 8
    mel = torch.zeros(B, 80, T_max)
    insts, trans = [], []
    for i, s in enumerate(samples):
        mel[i, :, : s.mel.shape[1]] = torch.from_numpy(s.mel)
        formatted = i >= B // 2
        insts.append(tokenizer.encode(INSTRUCTION_VARIANTS[int(formatted)]))
        trans.append(tokenizer.encode(s.ref_formatted if formatted else s.ref_unformatted))

    def _pad(lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.zeros(B, max(len(x) for x in lists), dtype=torch.long)
        for i, x in enumerate(lists):
            out[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        return out, torch.tensor([len(x) for x in lists], dtype=torch.long)

    inst_ids, inst_lens = _pad(insts)
    trans_ids, trans_lens = _pad(trans)
    return {
        "mel": mel,
        "audio_lengths": torch.tensor([s.audio_length for s in samples], dtype=torch.long),
        "instruction_ids": inst_ids, "instruction_lengths": inst_lens,
        "transcript_ids": trans_ids, "transcript_lengths": trans_lens,
    }


@torch.no_grad()
def _forward(encoder, adapter, llama, batch: dict, chat: ChatTemplate):
    """One teacher-forced forward pass plus a short greedy decode."""
    adapter_out = adapter(encoder(batch["mel"]))
    inputs, labels, audio_mask = prepare_input(
        adapter_out, batch["audio_lengths"],
        batch["instruction_ids"], batch["instruction_lengths"],
        batch["transcript_ids"], batch["transcript_lengths"],
        llama.embed_tokens, chat, adapter.audio_bos, adapter.audio_eos,
    )
    logits, loss = llama(inputs, labels, audio_mask=audio_mask)
    gen = greedy_generate(
        encoder, adapter, llama,
        batch["mel"], batch["audio_lengths"],
        batch["instruction_ids"], batch["instruction_lengths"],
        chat, max_new_tokens=_MAX_NEW_TOKENS,
    )
    return logits, loss, gen


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=_REPO / "configs" / "stub_chat.yaml")
    p.add_argument("--shard", type=Path, default=_REPO / "data_stub" / "stub-diag-000000.tar")
    p.add_argument("--out-dir", type=Path, default=_REPO / "out" / "equivalence")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--verify", action="store_true",
                   help="reload the saved checkpoint and compare against the capture")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if not cfg.model.stub:
        raise SystemExit(f"{args.config}: equivalence check runs on a stub config only")
    device = torch.device("cpu")
    torch.set_grad_enabled(False)

    _seed_all(args.seed)
    encoder, adapter, llama, _ = build_models(cfg, device, train=False, apply_init_from=False)
    modules = {"encoder": encoder, "adapter": adapter, "llama": llama}

    tokenizer = PrunedTokenizer(cfg.data.tokenizer)
    chat = ChatTemplate.from_tokenizer(tokenizer)
    batch = _build_batch(args.shard, tokenizer)

    ckpt_path = args.out_dir / "model.pt"
    ref_path = args.out_dir / "reference.pt"

    if not args.verify:
        touched = _fill_zero_tensors(modules, args.seed)
        print(f"[capture] filled {len(touched)} all-zero tensors, e.g. {touched[:3]}")
        logits, loss, gen = _forward(encoder, adapter, llama, batch, chat)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: m.state_dict() for k, m in modules.items()}, ckpt_path)
        torch.save({"batch": batch, "logits": logits, "loss": loss, "generated": gen,
                    "config": str(args.config), "seed": args.seed}, ref_path)
        print(f"[capture] logits {tuple(logits.shape)}  loss {loss.item():.6f}  "
              f"generated {gen}")
        print(f"[capture] wrote {ckpt_path} and {ref_path}")
        return 0

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ref = torch.load(ref_path, map_location="cpu", weights_only=True)
    if set(state) != set(modules):
        raise SystemExit(f"checkpoint modules {sorted(state)} != {sorted(modules)}")
    for name, mod in modules.items():
        mod.load_state_dict(state[name], strict=True)
    print("[verify] checkpoint loaded with strict=True into encoder, adapter, llama")

    for k, saved in ref["batch"].items():
        if not torch.equal(batch[k], saved):
            raise SystemExit(f"[verify] FAIL: rebuilt batch differs from capture in {k!r}")
    print("[verify] rebuilt batch identical to the captured one")

    logits, loss, gen = _forward(encoder, adapter, llama, ref["batch"], chat)
    max_diff = (logits - ref["logits"]).abs().max().item()
    loss_diff = abs(loss.item() - ref["loss"].item())
    print(f"[verify] max |Δlogits| {max_diff:.3e}  |Δloss| {loss_diff:.3e}  "
          f"generated equal: {gen == ref['generated']}")
    ok = (logits.shape == ref["logits"].shape
          and torch.allclose(logits, ref["logits"], atol=_ATOL, rtol=0)
          and loss_diff <= _ATOL
          and gen == ref["generated"])
    print("[verify] PASS" if ok else "[verify] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
