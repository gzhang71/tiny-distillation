"""Training-weight strategies for calibrated labels."""

from __future__ import annotations

from tiny_distillation.calibrated_labels.base import WeightingStrategy
from tiny_distillation.calibrated_labels.filtering import normalized_entropy
from tiny_distillation.core.math_utils import clamp
from tiny_distillation.core.types import ScoredPrediction


class _BoundedWeightingStrategy(WeightingStrategy):
    def __init__(self, minimum_weight: float = 0.1) -> None:
        if not 0 <= minimum_weight <= 1:
            raise ValueError("minimum_weight must be in [0, 1]")
        self.minimum_weight = minimum_weight

    def _bounded(self, value: float) -> float:
        return max(self.minimum_weight, clamp(value))


class ScoreWeighting(_BoundedWeightingStrategy):
    """Weights labels by the score stage's total quality score."""

    name = "score"

    def weight(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
    ) -> float:
        return self._bounded(item.total_score)


class ConfidenceWeighting(_BoundedWeightingStrategy):
    """Weights labels by maximum calibrated class probability."""

    name = "confidence"

    def weight(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
    ) -> float:
        return self._bounded(max(probabilities))


class EntropyWeighting(_BoundedWeightingStrategy):
    """Downweights uncertain, high-entropy teacher distributions."""

    name = "entropy"

    def weight(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
    ) -> float:
        return self._bounded(1.0 - normalized_entropy(probabilities))


class MarginWeighting(_BoundedWeightingStrategy):
    """Weights labels by the probability gap between the top two classes."""

    name = "margin"

    def weight(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
    ) -> float:
        ordered = sorted(probabilities, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        return self._bounded(margin)
