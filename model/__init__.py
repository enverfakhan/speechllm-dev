"""speech-llm model package.

Exports the three main model components, sequence utilities, and the audio
preprocessing utility.
"""

from model.whisper_encoder import WhisperEncoder, log_mel_spectrogram
from model.adapter import AudioSwiGLUBridge
from model.sequence import (
    ChatTemplate,
    EvalPrefixBatch,
    prepare_input,
)
from model.llama import Llama, LlamaConfig

__all__ = [
    "WhisperEncoder",
    "log_mel_spectrogram",
    "AudioSwiGLUBridge",
    "ChatTemplate",
    "EvalPrefixBatch",
    "prepare_input",
    "Llama",
    "LlamaConfig",
]
