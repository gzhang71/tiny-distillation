"""Batch orchestration for independently scoreable teacher traces."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from tiny_distillation.core.types import TeacherPrediction, TrainingExample
from tiny_distillation.generate_reasoning.base import ReasoningStrategy
from tiny_distillation.generate_reasoning.strategies import (
    ReasoningStrategyName,
    resolve_reasoning_strategy,
)
from tiny_distillation.teachers import Teacher


@dataclass(frozen=True)
class ReasoningGenerationConfig:
    strategy: ReasoningStrategy | ReasoningStrategyName | str = (
        ReasoningStrategyName.RATIONALE
    )
    candidates_per_example: int = 1
    include_reasoning: bool | None = None
    custom_instruction: str | None = None
    deduplicate_candidates: bool = False

    def __post_init__(self) -> None:
        if self.candidates_per_example < 1:
            raise ValueError("candidates_per_example must be at least 1")
        resolve_reasoning_strategy(self.strategy)


def generate_reasoning(
    examples: Iterable[TrainingExample],
    teacher: Teacher,
    config: ReasoningGenerationConfig | None = None,
    *,
    on_error: Callable[[TrainingExample, Exception], None] | None = None,
) -> list[TeacherPrediction]:
    """Generate one or more independently scoreable traces per example."""
    config = config or ReasoningGenerationConfig()
    strategy = resolve_reasoning_strategy(config.strategy)
    predictions: list[TeacherPrediction] = []
    seen: set[tuple[str, str, str]] = set()
    for example in examples:
        for candidate_index in range(config.candidates_per_example):
            try:
                prediction = strategy.generate_candidate(
                    example,
                    teacher,
                    candidate_index=candidate_index,
                    include_reasoning=config.include_reasoning,
                    custom_instruction=config.custom_instruction,
                )
            except Exception as exc:
                if on_error is None:
                    raise
                on_error(example, exc)
                continue
            candidate_key = (
                prediction.example_id,
                _normalize(prediction.answer),
                _normalize(prediction.reasoning),
            )
            if config.deduplicate_candidates and candidate_key in seen:
                continue
            seen.add(candidate_key)
            predictions.append(prediction)
    return predictions


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())
