"""Small, composable building blocks for language-model distillation."""

from tiny_distillation.core import (
    CalibratedLabel,
    ScoredPrediction,
    TeacherPrediction,
    TrainingExample,
)
from tiny_distillation.evaluation import EvaluationReport, evaluate_classification
from tiny_distillation.inference import (
    AutoregressiveModel,
    SpeculativeDecodingConfig,
    SpeculativeDecodingResult,
    speculative_decode,
)
from tiny_distillation.pipeline import DistillationPipeline, PipelineArtifacts
from tiny_distillation.preparation import (
    CalibrationConfig,
    CompositeScorer,
    LabelCalibrator,
    ReasoningGenerationConfig,
    ScoringConfig,
    generate_reasoning,
)
from tiny_distillation.teachers import CallableTeacher, RuleBasedArithmeticTeacher, Teacher
from tiny_distillation.training import (
    DistillationMode,
    StudentTrainer,
    TinyStudent,
    TrainerConfig,
    Vocabulary,
)

__all__ = [
    "CalibratedLabel",
    "CalibrationConfig",
    "CallableTeacher",
    "CompositeScorer",
    "DistillationMode",
    "DistillationPipeline",
    "EvaluationReport",
    "LabelCalibrator",
    "PipelineArtifacts",
    "ReasoningGenerationConfig",
    "RuleBasedArithmeticTeacher",
    "ScoredPrediction",
    "ScoringConfig",
    "SpeculativeDecodingConfig",
    "SpeculativeDecodingResult",
    "StudentTrainer",
    "Teacher",
    "TeacherPrediction",
    "TinyStudent",
    "TrainerConfig",
    "TrainingExample",
    "Vocabulary",
    "AutoregressiveModel",
    "evaluate_classification",
    "generate_reasoning",
    "speculative_decode",
]
