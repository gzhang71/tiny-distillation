"""Inference-time acceleration."""

from tiny_distillation.inference.speculative_decoding import (
    AutoregressiveModel,
    SpeculativeDecodingConfig,
    SpeculativeDecodingResult,
    speculative_decode,
)

__all__ = [
    "AutoregressiveModel",
    "SpeculativeDecodingConfig",
    "SpeculativeDecodingResult",
    "speculative_decode",
]
