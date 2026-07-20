"""Teacher-data generation, scoring, filtering, and calibration."""

from tiny_distillation.preparation.calibrated_labels import (
    CalibrationConfig,
    LabelCalibrator,
)
from tiny_distillation.preparation.generate_reasoning import (
    ReasoningGenerationConfig,
    generate_reasoning,
)
from tiny_distillation.preparation.score import (
    CompositeScorer,
    ScoringConfig,
    normalize_answer,
)

__all__ = [
    "CalibrationConfig",
    "CompositeScorer",
    "LabelCalibrator",
    "ReasoningGenerationConfig",
    "ScoringConfig",
    "generate_reasoning",
    "normalize_answer",
]
