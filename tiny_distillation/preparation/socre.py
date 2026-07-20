"""Compatibility import for the original requested module spelling."""

from tiny_distillation.preparation.score import (
    CompositeScorer,
    ScoringConfig,
    normalize_answer,
)

__all__ = ["CompositeScorer", "ScoringConfig", "normalize_answer"]
