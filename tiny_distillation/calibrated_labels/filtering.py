"""Quality filters for calibrated teacher labels."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tiny_distillation.calibrated_labels.base import LabelFilter
from tiny_distillation.core.types import ScoredPrediction


class AcceptedLabelFilter(LabelFilter):
    """Uses the acceptance decision from the scoring stage."""

    name = "accepted"

    def __init__(self, accepted_only: bool = True) -> None:
        self.accepted_only = accepted_only

    def keep(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
    ) -> bool:
        return item.accepted or not self.accepted_only


class QualityLabelFilter(LabelFilter):
    """Filters by confidence, normalized entropy, and top-class margin."""

    name = "quality"

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.0,
        maximum_entropy: float = 1.0,
        minimum_margin: float = 0.0,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if not 0 <= maximum_entropy <= 1:
            raise ValueError("maximum_entropy must be in [0, 1]")
        if not 0 <= minimum_margin <= 1:
            raise ValueError("minimum_margin must be in [0, 1]")
        self.minimum_confidence = minimum_confidence
        self.maximum_entropy = maximum_entropy
        self.minimum_margin = minimum_margin

    def keep(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
    ) -> bool:
        ordered = sorted(probabilities, reverse=True)
        confidence = ordered[0]
        margin = confidence - ordered[1] if len(ordered) > 1 else confidence
        entropy = normalized_entropy(probabilities)
        return (
            confidence >= self.minimum_confidence
            and entropy <= self.maximum_entropy
            and margin >= self.minimum_margin
        )


class CompositeLabelFilter(LabelFilter):
    """Keeps a label only when every component filter accepts it."""

    name = "composite"

    def __init__(self, filters: Iterable[LabelFilter]) -> None:
        self.filters = tuple(filters)

    def keep(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
    ) -> bool:
        return all(label_filter.keep(item, probabilities) for label_filter in self.filters)


def normalized_entropy(probabilities: tuple[float, ...]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(
        probability * math.log(max(probability, 1e-12))
        for probability in probabilities
    )
    return entropy / math.log(len(probabilities))
