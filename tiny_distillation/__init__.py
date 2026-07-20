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
from tiny_distillation.calibrated_labels import (
    CalibrationConfig,
    LabelCalibrator,
)
from tiny_distillation.generate_reasoning import (
    AnswerThenRationaleStrategy,
    CritiqueRevisionStrategy,
    DirectStrategy,
    RationaleStrategy,
    ReasoningGenerationConfig,
    ReasoningStrategy,
    ReasoningStrategyName,
    SelfConsistencyStrategy,
    StepByStepStrategy,
    generate_reasoning,
)
from tiny_distillation.score import (
    CompositeScorer,
    ScoringConfig,
)
from tiny_distillation.teachers import (
    CallableTeacher,
    ChatGPTTeacher,
    ClaudeTeacher,
    DeepSeekTeacher,
    LlamaTeacher,
    Qwen35Teacher,
    RuleBasedArithmeticTeacher,
    T5Teacher,
    Teacher,
    TeacherResponse,
)
from tiny_distillation.training import (
    DistillationMode,
    StudentTrainer,
    TinyStudent,
    TrainerConfig,
    Vocabulary,
)

__all__ = [
    "AnswerThenRationaleStrategy",
    "CalibratedLabel",
    "CalibrationConfig",
    "CallableTeacher",
    "ChatGPTTeacher",
    "ClaudeTeacher",
    "CompositeScorer",
    "CritiqueRevisionStrategy",
    "DirectStrategy",
    "DistillationMode",
    "DistillationPipeline",
    "DeepSeekTeacher",
    "EvaluationReport",
    "LabelCalibrator",
    "LlamaTeacher",
    "PipelineArtifacts",
    "Qwen35Teacher",
    "RationaleStrategy",
    "ReasoningGenerationConfig",
    "ReasoningStrategy",
    "ReasoningStrategyName",
    "RuleBasedArithmeticTeacher",
    "ScoredPrediction",
    "ScoringConfig",
    "SpeculativeDecodingConfig",
    "SpeculativeDecodingResult",
    "SelfConsistencyStrategy",
    "StepByStepStrategy",
    "StudentTrainer",
    "Teacher",
    "TeacherPrediction",
    "TeacherResponse",
    "T5Teacher",
    "TinyStudent",
    "TrainerConfig",
    "TrainingExample",
    "Vocabulary",
    "AutoregressiveModel",
    "evaluate_classification",
    "generate_reasoning",
    "speculative_decode",
]
