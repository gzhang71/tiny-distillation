"""Teacher trace generation."""

from tiny_distillation.generate_reasoning.base import ReasoningStrategy
from tiny_distillation.generate_reasoning.generator import (
    ReasoningGenerationConfig,
    generate_reasoning,
)
from tiny_distillation.generate_reasoning.strategies import (
    AnswerThenRationaleStrategy,
    CritiqueRevisionStrategy,
    DirectStrategy,
    RationaleStrategy,
    ReasoningStrategyName,
    SelfConsistencyStrategy,
    StepByStepStrategy,
    resolve_reasoning_strategy,
)

__all__ = [
    "AnswerThenRationaleStrategy",
    "CritiqueRevisionStrategy",
    "DirectStrategy",
    "RationaleStrategy",
    "ReasoningGenerationConfig",
    "ReasoningStrategy",
    "ReasoningStrategyName",
    "SelfConsistencyStrategy",
    "StepByStepStrategy",
    "generate_reasoning",
    "resolve_reasoning_strategy",
]
