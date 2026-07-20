"""Probability-calibration strategies."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from tiny_distillation.calibrated_labels.base import CalibrationStrategy
from tiny_distillation.core.math_utils import softmax
from tiny_distillation.core.types import ScoredPrediction, TrainingExample


class IdentityCalibration(CalibrationStrategy):
    """Applies softmax without fitting additional parameters."""

    name = "identity"

    def fit(
        self,
        scored: Iterable[ScoredPrediction],
        examples: Iterable[TrainingExample],
    ) -> None:
        return None

    def calibrate(self, logits: tuple[float, ...]) -> tuple[float, ...]:
        return softmax(logits)


class TemperatureCalibration(CalibrationStrategy):
    """Fits one temperature by minimizing weighted validation NLL."""

    name = "temperature"

    def __init__(
        self,
        temperature: float = 1.0,
        *,
        fit_temperature: bool = True,
        temperature_min: float = 0.25,
        temperature_max: float = 5.0,
        temperature_steps: int = 80,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if temperature_min <= 0 or temperature_max < temperature_min:
            raise ValueError("invalid temperature search range")
        if temperature_steps < 2:
            raise ValueError("temperature_steps must be at least 2")
        self._temperature = temperature
        self.fit_temperature = fit_temperature
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.temperature_steps = temperature_steps

    @property
    def temperature(self) -> float:
        return self._temperature

    def fit(
        self,
        scored: Iterable[ScoredPrediction],
        examples: Iterable[TrainingExample],
    ) -> None:
        if not self.fit_temperature:
            return
        labels = {example.id: example.label for example in examples}
        supervised = [
            item
            for item in scored
            if labels.get(item.prediction.example_id) is not None
        ]
        if not supervised:
            return
        self._temperature = min(
            self._temperature_candidates(),
            key=lambda value: self._nll(supervised, labels, value),
        )

    def calibrate(self, logits: tuple[float, ...]) -> tuple[float, ...]:
        return softmax(logits, self._temperature)

    def _temperature_candidates(self) -> list[float]:
        log_min = math.log(self.temperature_min)
        log_max = math.log(self.temperature_max)
        return [
            math.exp(
                log_min
                + index * (log_max - log_min) / (self.temperature_steps - 1)
            )
            for index in range(self.temperature_steps)
        ]

    @staticmethod
    def _nll(
        scored: list[ScoredPrediction],
        labels: Mapping[str, int | None],
        temperature: float,
    ) -> float:
        loss = 0.0
        total_weight = 0.0
        for item in scored:
            label = labels[item.prediction.example_id]
            if label is None or not 0 <= label < len(item.prediction.logits):
                continue
            probability = softmax(item.prediction.logits, temperature)[label]
            sample_weight = max(0.0, item.total_score)
            loss -= sample_weight * math.log(max(probability, 1e-12))
            total_weight += sample_weight
        return loss / max(total_weight, 1e-12)
