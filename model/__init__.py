"""speech-llm model package.

Exports the three main model components, sequence utilities, and the audio
preprocessing utility.
"""

from model.whisper_encoder import WhisperEncoder, log_mel_spectrogram
from model.adapter import AudioAdapter, AudioSwiGLUBridge, build_bridge_adapter
from model.sequence import (
    assemble_inputs,
    ChatTemplate,
    EvalPrefixBatch,
    prepare_input,
    prepare_input_chat,
)
from model.llama import Llama, LlamaConfig

__all__ = [
    "WhisperEncoder",
    "log_mel_spectrogram",
    "AudioAdapter",
    "AudioSwiGLUBridge",
    "build_bridge_adapter",
    "assemble_inputs",
    "ChatTemplate",
    "EvalPrefixBatch",
    "prepare_input",
    "prepare_input_chat",
    "Llama",
    "LlamaConfig",
]
