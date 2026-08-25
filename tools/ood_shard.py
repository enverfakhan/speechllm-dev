"""Shard-writing helpers shared by the three tools/prepare_ood_*.py builders.

Tools in this repo are deliberately self-contained and duplication between them
is fine — but the shard LAYOUT is not a per-tool choice.  Three copies of "which
members a sample has, and how the audio is encoded" is exactly the kind of
duplication that drifts silently: a reader groups tar members by key and takes
whatever it finds, so a builder that spelled a member differently would produce
shards that load, and score, and are wrong.  This module is that one contract,
and nothing else lives here.

Layout written (the house format of tools/preprocess.py, plus one member):

    {key}.mel.npy          float16 (80, T) — model/whisper_encoder.log_mel_spectrogram
    {key}.unformatted.txt  verbatim reference, UTF-8
    {key}.formatted.txt    cased/punctuated reference, UTF-8
    {key}.flac             16 kHz mono source audio          <- OOD addition

The .flac member exists so tools/run_whisper_control.py can run the paired
control system over byte-identical audio through Whisper's own front end.  A
control scored on a differently-cut segment is not a control.  Readers that
require only the three house members (data.build_sorted_eval_dataloader,
tools/make_eval_subset.py) ignore it; it costs roughly 10% of shard size.

Keys must not contain "." — every reader in this repo splits a member name on
the FIRST dot to recover (key, ext), so a dotted key is silently truncated.
add_sample enforces that rather than trusting each caller to remember it.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def load_audio_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode an audio blob (wav / flac / mp3) to a mono float32 array.

    Args:
        raw: encoded audio bytes

    Returns:
        (samples (T,) float32 in [-1, 1], sample_rate)
    """
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    return np.ascontiguousarray(data), int(sr)


def resample(audio: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample a mono float32 array to target_sr, via torchaudio.

    Mirrors tools/preprocess.py::_load_audio so an OOD mel is produced by the
    same front end as every LibriSpeech mel this project has ever trained on.

    Args:
        audio:     (T,) float32 mono
        sr:        source sample rate
        target_sr: destination sample rate (default 16 kHz)

    Returns:
        (T',) float32 mono at target_sr
    """
    if sr == target_sr:
        return audio

    import torch
    import torchaudio

    wav = torch.from_numpy(audio).unsqueeze(0)
    out = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return out.squeeze(0).numpy()


def encode_flac(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode a mono float32 array as 16-bit FLAC bytes."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


class ShardWriter:
    """Append samples to one .tar shard in the house format.

    The tar is opened lazily on the first sample, so a cap that ends up matching
    nothing leaves no empty shard behind to be mistaken for a valid eval set.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.n_samples = 0
        self._tar: tarfile.TarFile | None = None

    def _open(self) -> tarfile.TarFile:
        if self._tar is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._tar = tarfile.open(self.path, "w")
        return self._tar

    def _add(self, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        self._open().addfile(info, io.BytesIO(data))

    def add_sample(
        self,
        key: str,
        mel: np.ndarray,
        unformatted: str,
        formatted: str,
        audio: np.ndarray | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """Write one sample's members.

        Args:
            key:         sample id; must not contain "." (readers split on it)
            mel:         (80, T) float16 log-mel
            unformatted: verbatim reference text
            formatted:   cased/punctuated reference text
            audio:       optional (T,) float32 mono source audio → {key}.flac
            sample_rate: sample rate of `audio`
        """
        if "." in key:
            raise ValueError(
                f"Shard key {key!r} contains '.'; every reader in this repo splits "
                "member names on the first dot, which would truncate this key."
            )
        if mel.dtype != np.float16 or mel.ndim != 2 or mel.shape[0] != 80:
            raise ValueError(f"mel must be float16 (80, T); got {mel.dtype} {mel.shape}")

        buf = io.BytesIO()
        np.save(buf, mel)
        self._add(f"{key}.mel.npy", buf.getvalue())
        self._add(f"{key}.unformatted.txt", unformatted.encode("utf-8"))
        self._add(f"{key}.formatted.txt", formatted.encode("utf-8"))
        if audio is not None:
            self._add(f"{key}.flac", encode_flac(audio, sample_rate))
        self.n_samples += 1

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None


def read_shard(tar_path: Path) -> dict[str, dict[str, bytes]]:
    """Group every member of a shard by sample key → {ext: bytes}.

    Same key/ext split as data.build_sorted_eval_dataloader, so what this
    returns is exactly what the eval loader will see.
    """
    groups: dict[str, dict[str, bytes]] = {}
    with tarfile.open(Path(tar_path), "r") as tf:
        for member in tf.getmembers():
            dot = member.name.find(".")
            if dot < 0:
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            groups.setdefault(member.name[:dot], {})[member.name[dot + 1:]] = fobj.read()
    return groups


def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.tar"
        w = ShardWriter(path)
        audio = (np.sin(np.arange(16000) * 0.01) * 0.1).astype(np.float32)
        w.add_sample("a-0001", np.zeros((80, 100), np.float16), "hi there", "Hi there.", audio)
        w.add_sample("a-0002", np.ones((80, 50), np.float16), "bye", "Bye.", None)
        w.close()

        got = read_shard(path)
        assert set(got) == {"a-0001", "a-0002"}, got.keys()
        assert set(got["a-0001"]) == {"mel.npy", "unformatted.txt", "formatted.txt", "flac"}
        assert set(got["a-0002"]) == {"mel.npy", "unformatted.txt", "formatted.txt"}
        assert got["a-0001"]["unformatted.txt"].decode() == "hi there"
        back, sr = load_audio_bytes(got["a-0001"]["flac"])
        assert sr == SAMPLE_RATE and abs(len(back) - len(audio)) <= 1, (sr, len(back))
        assert np.abs(back - audio[: len(back)]).max() < 1e-3, "flac round trip lost the signal"
        print("  [OK] ShardWriter round trip: members, flac fidelity, optional audio")

        for bad_key, mel, why in (
            ("has.dot", np.zeros((80, 4), np.float16), "dotted key"),
            ("ok", np.zeros((80, 4), np.float32), "wrong mel dtype"),
            ("ok", np.zeros((40, 4), np.float16), "wrong mel bands"),
        ):
            try:
                ShardWriter(Path(tmp) / "x.tar").add_sample(bad_key, mel, "a", "b")
            except ValueError:
                continue
            raise AssertionError(f"add_sample accepted {why}")
        print("  [OK] add_sample rejects dotted keys and malformed mels")

        # Lazy open: a writer that never got a sample leaves no file behind.
        empty = ShardWriter(Path(tmp) / "empty.tar")
        empty.close()
        assert not (Path(tmp) / "empty.tar").exists(), "empty shard was created"
        print("  [OK] no empty shard left behind")

    # resample: 48 kHz → 16 kHz thirds the length
    out = resample(np.zeros(48000, np.float32), 48000)
    assert abs(len(out) - 16000) < 10, len(out)
    assert resample(audio, SAMPLE_RATE) is audio, "no-op resample copied"
    print("  [OK] resample 48k→16k")
    print("PASSED")


if __name__ == "__main__":
    _self_test()
