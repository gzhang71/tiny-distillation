"""Batch generation and candidate selection for teacher traces."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable

from tiny_distillation.core.types import TeacherPrediction, TrainingExample
from tiny_distillation.teachers.teacher import Teacher


@dataclass(frozen=True)
class ReasoningGenerationConfig:
    candidates_per_example: int = 1
    include_reasoning: bool = True

    def __post_init__(self) -> None:
        if self.candidates_per_example < 1:
            raise ValueError("candidates_per_example must be at least 1")


def generate_reasoning(
    examples: Iterable[TrainingExample],
    teacher: Teacher,
    config: ReasoningGenerationConfig | None = None,
    *,
    on_error: Callable[[TrainingExample, Exception], None] | None = None,
) -> list[TeacherPrediction]:
    """Generate one or more independently scoreable traces per example."""
    config = config or ReasoningGenerationConfig()
    predictions: list[TeacherPrediction] = []
    for example in examples:
        for candidate_index in range(config.candidates_per_example):
            try:
                prediction = teacher.generate(
                    example,
                    include_reasoning=config.include_reasoning,
                    candidate_index=candidate_index,
                )
            except Exception as exc:
                if on_error is None:
                    raise
                on_error(example, exc)
                continue
            predictions.append(prediction)
    return predictions
