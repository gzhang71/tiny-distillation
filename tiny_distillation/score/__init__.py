"""Teacher output scoring and filtering."""

from tiny_distillation.score.scorer import (
    CompositeScorer,
    ScoringConfig,
    normalize_answer,
)

__all__ = ["CompositeScorer", "ScoringConfig", "normalize_answer"]
