"""Temperature calibration and conversion to trainable targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterable, Mapping

from tiny_distillation.core.math_utils import argmax, clamp, softmax
from tiny_distillation.core.types import (
    CalibratedLabel,
    ScoredPrediction,
    TrainingExample,
)


@dataclass(frozen=True)
class CalibrationConfig:
    temperature: float = 1.0
    fit_temperature: bool = True
    temperature_min: float = 0.25
    temperature_max: float = 5.0
    temperature_steps: int = 80
    minimum_weight: float = 0.10
    accepted_only: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.temperature_min <= 0 or self.temperature_max < self.temperature_min:
            raise ValueError("invalid temperature search range")
        if self.temperature_steps < 2:
            raise ValueError("temperature_steps must be at least 2")
        if not 0 <= self.minimum_weight <= 1:
            raise ValueError("minimum_weight must be in [0, 1]")


class LabelCalibrator:
    """Fits scalar temperature and emits hard, soft, and CoT targets together."""

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self.temperature = self.config.temperature

    def fit(
        self,
        scored: Iterable[ScoredPrediction],
        examples: Iterable[TrainingExample],
    ) -> float:
        if not self.config.fit_temperature:
            return self.temperature
        labels = {example.id: example.label for example in examples}
        supervised = [
            item
            for item in scored
            if labels.get(item.prediction.example_id) is not None
        ]
        if not supervised:
            return self.temperature
        candidates = self._temperature_candidates()
        self.temperature = min(
            candidates,
            key=lambda temperature: self._nll(supervised, labels, temperature),
        )
        return self.temperature

    def transform(
        self,
        scored: Iterable[ScoredPrediction],
    ) -> list[CalibratedLabel]:
        labels: list[CalibratedLabel] = []
        for item in scored:
            if self.config.accepted_only and not item.accepted:
                continue
            probabilities = softmax(item.prediction.logits, self.temperature)
            labels.append(
                CalibratedLabel(
                    example_id=item.prediction.example_id,
                    prompt=item.prediction.prompt,
                    hard_label=argmax(probabilities),
                    soft_labels=probabilities,
                    answer=item.prediction.answer,
                    reasoning=item.prediction.reasoning,
                    weight=max(self.config.minimum_weight, clamp(item.total_score)),
                    source_score=item.total_score,
                )
            )
        return labels

    def fit_transform(
        self,
        scored: Iterable[ScoredPrediction],
        examples: Iterable[TrainingExample],
    ) -> list[CalibratedLabel]:
        scored_items = list(scored)
        self.fit(scored_items, examples)
        return self.transform(scored_items)

    def _temperature_candidates(self) -> list[float]:
        config = self.config
        log_min = math.log(config.temperature_min)
        log_max = math.log(config.temperature_max)
        return [
            math.exp(log_min + index * (log_max - log_min) / (config.temperature_steps - 1))
            for index in range(config.temperature_steps)
        ]

    @staticmethod
    def _nll(
        scored: list[ScoredPrediction],
        labels: Mapping[str, int | None],
        temperature: float,
    ) -> float:
        loss = 0.0
        weight = 0.0
        for item in scored:
            label = labels[item.prediction.example_id]
            if label is None or not 0 <= label < len(item.prediction.logits):
                continue
            probability = softmax(item.prediction.logits, temperature)[label]
            loss -= item.total_score * math.log(max(probability, 1e-12))
            weight += item.total_score
        return loss / max(weight, 1e-12)
