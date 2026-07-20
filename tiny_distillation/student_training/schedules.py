"""Loss-weight schedules for student training."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tiny_distillation.student_training.losses import LossWeights


class LossWeightScheduler(ABC):
    """Returns objective weights for one epoch."""

    @abstractmethod
    def weights(
        self,
        epoch: int,
        total_epochs: int,
        base_weights: LossWeights,
    ) -> LossWeights:
        ...


class StaticLossWeightScheduler(LossWeightScheduler):
    def weights(
        self,
        epoch: int,
        total_epochs: int,
        base_weights: LossWeights,
    ) -> LossWeights:
        return base_weights


class LinearCoTCurriculum(LossWeightScheduler):
    """Linearly introduces rationale loss over an initial epoch window."""

    def __init__(self, warmup_epochs: int) -> None:
        if warmup_epochs < 1:
            raise ValueError("warmup_epochs must be positive")
        self.warmup_epochs = warmup_epochs

    def weights(
        self,
        epoch: int,
        total_epochs: int,
        base_weights: LossWeights,
    ) -> LossWeights:
        progress = min(1.0, (epoch + 1) / self.warmup_epochs)
        return LossWeights(
            hard=base_weights.hard,
            soft=base_weights.soft,
            cot=base_weights.cot * progress,
        )
