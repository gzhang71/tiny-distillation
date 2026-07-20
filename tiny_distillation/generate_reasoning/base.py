"""Base class for configurable reasoning-generation behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from tiny_distillation.core.types import TeacherPrediction, TrainingExample
from tiny_distillation.teachers import Teacher


class ReasoningStrategy(ABC):
    """Produces one independently scoreable teacher candidate."""

    name = "base"
    default_include_reasoning = True

    def generate_candidate(
        self,
        example: TrainingExample,
        teacher: Teacher,
        candidate_index: int,
        *,
        include_reasoning: bool | None = None,
        custom_instruction: str | None = None,
    ) -> TeacherPrediction:
        instruction = self.build_instruction(example, candidate_index)
        if custom_instruction:
            instruction = f"{instruction}\n{custom_instruction}"
        return self._invoke_teacher(
            example,
            teacher,
            candidate_index,
            instruction=instruction,
            include_reasoning=include_reasoning,
        )

    @abstractmethod
    def build_instruction(
        self,
        example: TrainingExample,
        candidate_index: int,
    ) -> str:
        """Return the behavior instruction for one candidate."""

    def _invoke_teacher(
        self,
        example: TrainingExample,
        teacher: Teacher,
        candidate_index: int,
        *,
        instruction: str,
        include_reasoning: bool | None,
    ) -> TeacherPrediction:
        should_include_reasoning = (
            self.default_include_reasoning
            if include_reasoning is None
            else include_reasoning
        )
        strategy_example = replace(
            example,
            metadata={
                **dict(example.metadata),
                "generation_instruction": instruction,
                "reasoning_strategy": self.name,
            },
        )
        prediction = teacher.generate(
            strategy_example,
            include_reasoning=should_include_reasoning,
            candidate_index=candidate_index,
        )
        return replace(
            prediction,
            example_id=example.id,
            prompt=example.prompt,
            metadata={
                **dict(prediction.metadata),
                "reasoning_strategy": self.name,
                "generation_instruction": instruction,
            },
        )

