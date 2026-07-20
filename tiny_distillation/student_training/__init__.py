"""Student architecture and distillation training."""

from tiny_distillation.student_training.losses import (
    CoTTokenCrossEntropyLoss,
    CombinedDistillationLoss,
    DistillationLoss,
    DistillationLossOutput,
    HardCrossEntropyLoss,
    LossWeights,
    SoftKLDistillationLoss,
)
from tiny_distillation.student_training.models import (
    StudentModel,
    TinyStudent,
    TransformerStudent,
)
from tiny_distillation.student_training.schedules import (
    LinearCoTCurriculum,
    LossWeightScheduler,
    StaticLossWeightScheduler,
)
from tiny_distillation.student_training.student_training import (
    DistillationMode,
    SamplingMethod,
    StudentTrainer,
    TrainerConfig,
    Vocabulary,
)

__all__ = [
    "CoTTokenCrossEntropyLoss",
    "CombinedDistillationLoss",
    "DistillationLoss",
    "DistillationLossOutput",
    "DistillationMode",
    "HardCrossEntropyLoss",
    "LinearCoTCurriculum",
    "LossWeightScheduler",
    "LossWeights",
    "SamplingMethod",
    "SoftKLDistillationLoss",
    "StaticLossWeightScheduler",
    "StudentModel",
    "StudentTrainer",
    "TinyStudent",
    "TrainerConfig",
    "TransformerStudent",
    "Vocabulary",
]
