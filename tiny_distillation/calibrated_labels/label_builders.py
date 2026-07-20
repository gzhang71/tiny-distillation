"""Hard- and soft-target construction strategies."""

from __future__ import annotations

from tiny_distillation.calibrated_labels.base import LabelBuilder, LabelTargets
from tiny_distillation.core.math_utils import argmax
from tiny_distillation.core.types import ScoredPrediction, TrainingExample


class TeacherLabelBuilder(LabelBuilder):
    """Uses the calibrated teacher distribution as supervision."""

    name = "teacher"

    def __init__(
        self,
        *,
        label_smoothing: float = 0.0,
        top_k: int | None = None,
    ) -> None:
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        self.label_smoothing = label_smoothing
        self.top_k = top_k

    def build(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
        example: TrainingExample | None,
    ) -> LabelTargets:
        soft_labels = _top_k_distribution(probabilities, self.top_k)
        soft_labels = _smooth_distribution(soft_labels, self.label_smoothing)
        return LabelTargets(
            hard_label=argmax(probabilities),
            soft_labels=soft_labels,
        )


class GroundTruthBlendLabelBuilder(LabelBuilder):
    """Blends teacher probabilities with a known one-hot target."""

    name = "ground_truth_blend"

    def __init__(
        self,
        *,
        ground_truth_weight: float = 0.5,
        label_smoothing: float = 0.0,
        top_k: int | None = None,
    ) -> None:
        if not 0 <= ground_truth_weight <= 1:
            raise ValueError("ground_truth_weight must be in [0, 1]")
        self.ground_truth_weight = ground_truth_weight
        self.teacher_builder = TeacherLabelBuilder(
            label_smoothing=label_smoothing,
            top_k=top_k,
        )

    def build(
        self,
        item: ScoredPrediction,
        probabilities: tuple[float, ...],
        example: TrainingExample | None,
    ) -> LabelTargets:
        teacher_targets = self.teacher_builder.build(item, probabilities, example)
        if (
            example is None
            or example.label is None
            or not 0 <= example.label < len(probabilities)
        ):
            return teacher_targets
        teacher_weight = 1.0 - self.ground_truth_weight
        soft_labels = tuple(
            teacher_weight * probability
            + self.ground_truth_weight * float(index == example.label)
            for index, probability in enumerate(teacher_targets.soft_labels)
        )
        return LabelTargets(
            hard_label=example.label,
            soft_labels=soft_labels,
        )


def _top_k_distribution(
    probabilities: tuple[float, ...],
    top_k: int | None,
) -> tuple[float, ...]:
    if top_k is None or top_k >= len(probabilities):
        return probabilities
    retained = set(
        sorted(
            range(len(probabilities)),
            key=probabilities.__getitem__,
            reverse=True,
        )[:top_k]
    )
    total = sum(
        probability
        for index, probability in enumerate(probabilities)
        if index in retained
    )
    return tuple(
        probability / total if index in retained else 0.0
        for index, probability in enumerate(probabilities)
    )


def _smooth_distribution(
    probabilities: tuple[float, ...],
    smoothing: float,
) -> tuple[float, ...]:
    if smoothing == 0:
        return probabilities
    uniform = smoothing / len(probabilities)
    return tuple(
        (1.0 - smoothing) * probability + uniform
        for probability in probabilities
    )
