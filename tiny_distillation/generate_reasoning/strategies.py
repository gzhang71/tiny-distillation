"""Built-in reasoning-generation strategies."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

from tiny_distillation.core.types import TeacherPrediction, TrainingExample
from tiny_distillation.generate_reasoning.base import ReasoningStrategy
from tiny_distillation.teachers import Teacher


class ReasoningStrategyName(str, Enum):
    DIRECT = "direct"
    RATIONALE = "rationale"
    STEP_BY_STEP = "step_by_step"
    ANSWER_THEN_RATIONALE = "answer_then_rationale"
    CRITIQUE_REVISION = "critique_revision"
    SELF_CONSISTENCY = "self_consistency"


class DirectStrategy(ReasoningStrategy):
    name = ReasoningStrategyName.DIRECT.value
    default_include_reasoning = False

    def build_instruction(
        self,
        example: TrainingExample,
        candidate_index: int,
    ) -> str:
        return "Return the final answer directly and leave the reasoning field empty."


class RationaleStrategy(ReasoningStrategy):
    name = ReasoningStrategyName.RATIONALE.value

    def build_instruction(
        self,
        example: TrainingExample,
        candidate_index: int,
    ) -> str:
        return (
            "Give a concise, self-contained rationale that supports the final answer. "
            "Include only reasoning useful as student supervision."
        )


class StepByStepStrategy(ReasoningStrategy):
    name = ReasoningStrategyName.STEP_BY_STEP.value

    def build_instruction(
        self,
        example: TrainingExample,
        candidate_index: int,
    ) -> str:
        return (
            "Explain the solution as a short sequence of numbered, verifiable steps. "
            "End the reasoning with a check that is consistent with the final answer."
        )


class AnswerThenRationaleStrategy(ReasoningStrategy):
    name = ReasoningStrategyName.ANSWER_THEN_RATIONALE.value

    def build_instruction(
        self,
        example: TrainingExample,
        candidate_index: int,
    ) -> str:
        return (
            "Decide the final answer first. Then provide a concise rationale that "
            "supports that fixed answer without changing it."
        )


class SelfConsistencyStrategy(ReasoningStrategy):
    name = ReasoningStrategyName.SELF_CONSISTENCY.value

    def build_instruction(
        self,
        example: TrainingExample,
        candidate_index: int,
    ) -> str:
        return (
            f"Produce independent solution candidate {candidate_index + 1}. "
            "Solve from the original question without assuming another candidate's "
            "answer or reasoning."
        )


class CritiqueRevisionStrategy(ReasoningStrategy):
    name = ReasoningStrategyName.CRITIQUE_REVISION.value

    def build_instruction(
        self,
        example: TrainingExample,
        candidate_index: int,
    ) -> str:
        return (
            "Create a draft answer and concise rationale that can be checked by a "
            "second teacher pass."
        )

    def generate_candidate(
        self,
        example: TrainingExample,
        teacher: Teacher,
        candidate_index: int,
        *,
        include_reasoning: bool | None = None,
        custom_instruction: str | None = None,
    ) -> TeacherPrediction:
        draft_instruction = self.build_instruction(example, candidate_index)
        draft = self._invoke_teacher(
            example,
            teacher,
            candidate_index,
            instruction=draft_instruction,
            include_reasoning=True,
        )
        revision_instruction = (
            "Critique the draft below for correctness, relevance, and internal "
            "consistency. Return a corrected final answer and concise revised "
            "rationale. Do not discuss the review process in the final rationale.\n"
            f"Draft answer: {draft.answer}\n"
            f"Draft rationale: {draft.reasoning}"
        )
        if custom_instruction:
            revision_instruction = f"{revision_instruction}\n{custom_instruction}"
        revised = self._invoke_teacher(
            example,
            teacher,
            candidate_index,
            instruction=revision_instruction,
            include_reasoning=include_reasoning,
        )
        return replace(
            revised,
            metadata={
                **dict(revised.metadata),
                "draft_answer": draft.answer,
                "draft_reasoning": draft.reasoning,
            },
        )


_BUILT_IN_STRATEGIES: dict[ReasoningStrategyName, ReasoningStrategy] = {
    ReasoningStrategyName.DIRECT: DirectStrategy(),
    ReasoningStrategyName.RATIONALE: RationaleStrategy(),
    ReasoningStrategyName.STEP_BY_STEP: StepByStepStrategy(),
    ReasoningStrategyName.ANSWER_THEN_RATIONALE: AnswerThenRationaleStrategy(),
    ReasoningStrategyName.CRITIQUE_REVISION: CritiqueRevisionStrategy(),
    ReasoningStrategyName.SELF_CONSISTENCY: SelfConsistencyStrategy(),
}


def resolve_reasoning_strategy(
    strategy: ReasoningStrategy | ReasoningStrategyName | str,
) -> ReasoningStrategy:
    if isinstance(strategy, ReasoningStrategy):
        return strategy
    try:
        name = ReasoningStrategyName(strategy)
    except ValueError as exc:
        choices = ", ".join(item.value for item in ReasoningStrategyName)
        raise ValueError(f"unknown reasoning strategy {strategy!r}; choose {choices}") from exc
    return _BUILT_IN_STRATEGIES[name]

