"""Shared interface for teacher-output scoring strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from tiny_distillation.core.types import (
    ScoredPrediction,
    TeacherPrediction,
    TrainingExample,
)


class ScoringStrategy(ABC):
    """Scores teacher predictions and selects the best candidate per example."""

    @abstractmethod
    def score(
        self,
        predictions: Iterable[TeacherPrediction],
        examples: Iterable[TrainingExample],
    ) -> list[ScoredPrediction]:
        ...

    @staticmethod
    def best_per_example(
        scored: Iterable[ScoredPrediction],
    ) -> list[ScoredPrediction]:
        best: dict[str, ScoredPrediction] = {}
        for item in scored:
            example_id = item.prediction.example_id
            if example_id not in best or item.total_score > best[example_id].total_score:
                best[example_id] = item
        return list(best.values())

