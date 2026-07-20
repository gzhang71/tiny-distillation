"""Quality scoring and filtering for teacher generations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Callable, Iterable

from tiny_distillation.core.math_utils import clamp
from tiny_distillation.core.types import (
    ScoredPrediction,
    TeacherPrediction,
    TrainingExample,
)


def normalize_answer(answer: str) -> str:
    return " ".join(answer.strip().lower().split())


@dataclass(frozen=True)
class ScoringConfig:
    answer_weight: float = 0.55
    reasoning_weight: float = 0.25
    confidence_weight: float = 0.20
    acceptance_threshold: float = 0.60
    minimum_reasoning_words: int = 4

    def __post_init__(self) -> None:
        weights = (self.answer_weight, self.reasoning_weight, self.confidence_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("scoring weights must be non-negative with a positive sum")
        if not 0 <= self.acceptance_threshold <= 1:
            raise ValueError("acceptance_threshold must be in [0, 1]")
        if self.minimum_reasoning_words < 1:
            raise ValueError("minimum_reasoning_words must be positive")


class CompositeScorer:
    """Combines correctness, trace quality, confidence, and an optional reward."""

    def __init__(
        self,
        config: ScoringConfig | None = None,
        reward_fn: Callable[[TeacherPrediction, TrainingExample], float] | None = None,
    ) -> None:
        self.config = config or ScoringConfig()
        self.reward_fn = reward_fn

    def score(
        self,
        predictions: Iterable[TeacherPrediction],
        examples: Iterable[TrainingExample],
    ) -> list[ScoredPrediction]:
        examples_by_id = {example.id: example for example in examples}
        return [
            self.score_one(prediction, examples_by_id[prediction.example_id])
            for prediction in predictions
        ]

    def score_one(
        self,
        prediction: TeacherPrediction,
        example: TrainingExample,
    ) -> ScoredPrediction:
        answer_score = self._answer_score(prediction, example)
        reasoning_score = self._reasoning_score(prediction)
        confidence_score = clamp(prediction.confidence)
        config = self.config
        weighted_sum = (
            config.answer_weight * answer_score
            + config.reasoning_weight * reasoning_score
            + config.confidence_weight * confidence_score
        )
        weight_total = (
            config.answer_weight + config.reasoning_weight + config.confidence_weight
        )
        total_score = weighted_sum / weight_total
        diagnostics: dict[str, float | str | bool] = {}
        if self.reward_fn is not None:
            reward = clamp(float(self.reward_fn(prediction, example)))
            total_score = 0.8 * total_score + 0.2 * reward
            diagnostics["external_reward"] = reward
        total_score = clamp(total_score)
        return ScoredPrediction(
            prediction=prediction,
            answer_score=answer_score,
            reasoning_score=reasoning_score,
            confidence_score=confidence_score,
            total_score=total_score,
            accepted=total_score >= config.acceptance_threshold,
            diagnostics=diagnostics,
        )

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

    @staticmethod
    def _answer_score(
        prediction: TeacherPrediction,
        example: TrainingExample,
    ) -> float:
        if example.reference_answer is None:
            return 1.0
        return float(
            normalize_answer(prediction.answer)
            == normalize_answer(example.reference_answer)
        )

    def _reasoning_score(self, prediction: TeacherPrediction) -> float:
        words = re.findall(r"\b[\w.+*-]+\b", prediction.reasoning)
        if not words:
            return 0.0
        length_score = min(1.0, len(words) / self.config.minimum_reasoning_words)
        answer_mentioned = normalize_answer(prediction.answer) in normalize_answer(
            prediction.reasoning
        )
        return 0.7 * length_score + 0.3 * float(answer_mentioned)
