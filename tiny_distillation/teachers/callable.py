"""Adapter for applications that already own the teacher invocation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from tiny_distillation.core.types import TeacherPrediction, TrainingExample
from tiny_distillation.teachers.base import Teacher, TeacherResponse


class CallableTeacher(Teacher):
    """Preserves the original callback-based Teacher adapter."""

    provider = "callable"

    def __init__(
        self,
        generate_fn: Callable[[TrainingExample, bool, int], TeacherPrediction],
    ) -> None:
        super().__init__(model="callable", labels=("<callback-owned>",))
        self._generate_fn = generate_fn

    def generate(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool = True,
        candidate_index: int = 0,
    ) -> TeacherPrediction:
        prediction = self._generate_fn(example, include_reasoning, candidate_index)
        if prediction.example_id != example.id or prediction.prompt != example.prompt:
            prediction = replace(
                prediction,
                example_id=example.id,
                prompt=example.prompt,
            )
        return prediction

    def _request(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool,
        candidate_index: int,
    ) -> TeacherResponse:
        raise RuntimeError("CallableTeacher delegates directly to generate_fn")

