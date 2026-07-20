"""Teacher output scoring and filtering."""

from tiny_distillation.score.base import ScoringStrategy
from tiny_distillation.score.scorer import (
    CompositeScorer,
    ScoringConfig,
    normalize_answer,
)
from tiny_distillation.score.strategies import (
    ConfidenceScorer,
    ConsensusScorer,
    ConsensusScoringConfig,
    ExactAnswerScorer,
    ReasoningQualityScorer,
    RewardScorer,
)

__all__ = [
    "CompositeScorer",
    "ConfidenceScorer",
    "ConsensusScorer",
    "ConsensusScoringConfig",
    "ExactAnswerScorer",
    "ReasoningQualityScorer",
    "RewardScorer",
    "ScoringConfig",
    "ScoringStrategy",
    "normalize_answer",
]
