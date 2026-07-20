"""Specialized scoring strategies for different distillation experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from tiny_distillation.core.math_utils import clamp
from tiny_distillation.core.types import (
    ScoredPrediction,
    TeacherPrediction,
    TrainingExample,
)
from tiny_distillation.score.base import ScoringStrategy
from tiny_distillation.score.scorer import (
    CompositeScorer,
    ScoringConfig,
    normalize_answer,
)


class _ComponentScorer(ScoringStrategy):
    def __init__(self, config: ScoringConfig) -> None:
        self._composite = CompositeScorer(config)

    def score(
        self,
        predictions: Iterable[TeacherPrediction],
        examples: Iterable[TrainingExample],
    ) -> list[ScoredPrediction]:
        return self._composite.score(predictions, examples)


class ExactAnswerScorer(_ComponentScorer):
    """Scores only exact agreement with the reference answer."""

    def __init__(self, acceptance_threshold: float = 1.0) -> None:
        super().__init__(
            ScoringConfig(
                answer_weight=1.0,
                reasoning_weight=0.0,
                confidence_weight=0.0,
                acceptance_threshold=acceptance_threshold,
            )
        )


class ConfidenceScorer(_ComponentScorer):
    """Scores only the teacher-reported or projected confidence."""

    def __init__(self, acceptance_threshold: float = 0.8) -> None:
        super().__init__(
            ScoringConfig(
                answer_weight=0.0,
                reasoning_weight=0.0,
                confidence_weight=1.0,
                acceptance_threshold=acceptance_threshold,
            )
        )


class ReasoningQualityScorer(_ComponentScorer):
    """Scores rationale length and whether it supports the answer."""

    def __init__(
        self,
        acceptance_threshold: float = 0.7,
        *,
        minimum_reasoning_words: int = 4,
    ) -> None:
        super().__init__(
            ScoringConfig(
                answer_weight=0.0,
                reasoning_weight=1.0,
                confidence_weight=0.0,
                acceptance_threshold=acceptance_threshold,
                minimum_reasoning_words=minimum_reasoning_words,
            )
        )


class RewardScorer(ScoringStrategy):
    """Uses a task-specific verifier or reward model as the total score."""

    def __init__(
        self,
        reward_fn: Callable[[TeacherPrediction, TrainingExample], float],
        *,
        acceptance_threshold: float = 0.6,
    ) -> None:
        if not 0 <= acceptance_threshold <= 1:
            raise ValueError("acceptance_threshold must be in [0, 1]")
        self.reward_fn = reward_fn
        self.acceptance_threshold = acceptance_threshold
        self._diagnostic_scorer = CompositeScorer(
            ScoringConfig(acceptance_threshold=0.0)
        )

    def score(
        self,
        predictions: Iterable[TeacherPrediction],
        examples: Iterable[TrainingExample],
    ) -> list[ScoredPrediction]:
        prediction_items = list(predictions)
        example_items = list(examples)
        examples_by_id = {example.id: example for example in example_items}
        diagnostics = self._diagnostic_scorer.score(
            prediction_items,
            example_items,
        )
        results: list[ScoredPrediction] = []
        for item in diagnostics:
            reward = clamp(
                float(
                    self.reward_fn(
                        item.prediction,
                        examples_by_id[item.prediction.example_id],
                    )
                )
            )
            results.append(
                replace(
                    item,
                    total_score=reward,
                    accepted=reward >= self.acceptance_threshold,
                    diagnostics={**dict(item.diagnostics), "external_reward": reward},
                )
            )
        return results


@dataclass(frozen=True)
class ConsensusScoringConfig:
    consensus_weight: float = 0.4
    acceptance_threshold: float = 0.6
    minimum_candidates: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.consensus_weight <= 1:
            raise ValueError("consensus_weight must be in [0, 1]")
        if not 0 <= self.acceptance_threshold <= 1:
            raise ValueError("acceptance_threshold must be in [0, 1]")
        if self.minimum_candidates < 2:
            raise ValueError("minimum_candidates must be at least 2")


class ConsensusScorer(ScoringStrategy):
    """Blends candidate quality with answer agreement for self-consistency."""

    def __init__(
        self,
        base_scorer: ScoringStrategy | None = None,
        config: ConsensusScoringConfig | None = None,
    ) -> None:
        self.base_scorer = base_scorer or CompositeScorer()
        self.config = config or ConsensusScoringConfig()

    def score(
        self,
        predictions: Iterable[TeacherPrediction],
        examples: Iterable[TrainingExample],
    ) -> list[ScoredPrediction]:
        prediction_items = list(predictions)
        scored = self.base_scorer.score(prediction_items, examples)
        answer_counts: dict[str, Counter[str]] = defaultdict(Counter)
        candidate_counts: Counter[str] = Counter()
        for prediction in prediction_items:
            answer_counts[prediction.example_id][
                normalize_answer(prediction.answer)
            ] += 1
            candidate_counts[prediction.example_id] += 1

        results: list[ScoredPrediction] = []
        for item in scored:
            example_id = item.prediction.example_id
            candidate_count = candidate_counts[example_id]
            if candidate_count < self.config.minimum_candidates:
                consensus_score = 0.0
            else:
                consensus_score = (
                    answer_counts[example_id][
                        normalize_answer(item.prediction.answer)
                    ]
                    / candidate_count
                )
            total_score = clamp(
                (1.0 - self.config.consensus_weight) * item.total_score
                + self.config.consensus_weight * consensus_score
            )
            results.append(
                replace(
                    item,
                    total_score=total_score,
                    accepted=total_score >= self.config.acceptance_threshold,
                    diagnostics={
                        **dict(item.diagnostics),
                        "base_score": item.total_score,
                        "consensus_score": consensus_score,
                        "candidate_count": float(candidate_count),
                    },
                )
            )
        return results

