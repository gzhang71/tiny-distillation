"""Shared interfaces for calibrated-label construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from tiny_distillation.core.types import ScoredPrediction, TrainingExample


@dataclass(frozen=True)
class LabelTargets:
    """Hard and soft supervision produced for one scored prediction."""

    hard_label: int
    soft_labels: tuple[float, ...]


class CalibrationStrategy(ABC):
    """Fits and applies a probability calibration method."""

    name = "base"

    @abstractmethod
    def fit(
        self,
        scored: Iterable[ScoredPrediction],
        examples: Iterable[TrainingExample],
    ) -> None:
        """Fit calibration parameters from labeled examples."""

    @abstractmethod
    def calibrate(self, logits: tuple[float, ...]) -> tuple[float, ...]:
        """Convert teacher logits into calibrated probabilities."""

    @property
    def temperature(self) -> float:
        """Scalar temperature exposed for pipeline compatibility."""

        return 1.0


class LabelBuilder(ABC):
    """Constructs hard and soft student targets."""

    name = "base"

    @abstractmethod
    def build(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
        example: TrainingExample | None,
    ) -> LabelTargets:
        """Build targets for one teacher prediction."""


class LabelFilter(ABC):
    """Decides whether a calibrated teacher prediction is retained."""

    name = "base"

    @abstractmethod
    def keep(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
    ) -> bool:
        """Return whether the prediction should become a training label."""


class WeightingStrategy(ABC):
    """Assigns a normalized training weight to a retained prediction."""

    name = "base"

    @abstractmethod
    def weight(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
    ) -> float:
        """Return a weight in the closed interval [0, 1]."""
