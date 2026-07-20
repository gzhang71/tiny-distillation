"""Teacher model interfaces and lightweight adapters."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from tiny_distillation.core.math_utils import softmax
from tiny_distillation.core.types import TeacherPrediction, TrainingExample


@runtime_checkable
class Teacher(Protocol):
    """Anything that can produce answer, reasoning, and label logits."""

    def generate(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool = True,
        candidate_index: int = 0,
    ) -> TeacherPrediction:
        ...


class CallableTeacher:
    """Adapts an SDK call or local model function to the Teacher protocol."""

    def __init__(
        self,
        generate_fn: Callable[[TrainingExample, bool, int], TeacherPrediction],
    ) -> None:
        self._generate_fn = generate_fn

    def generate(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool = True,
        candidate_index: int = 0,
    ) -> TeacherPrediction:
        prediction = self._generate_fn(example, include_reasoning, candidate_index)
        if prediction.example_id != example.id:
            prediction = replace(prediction, example_id=example.id, prompt=example.prompt)
        return prediction


class RuleBasedArithmeticTeacher:
    """Deterministic teacher used by the demo and offline tests."""

    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
    }

    def __init__(self, labels: Sequence[int]) -> None:
        if not labels:
            raise ValueError("labels cannot be empty")
        self.labels = tuple(labels)

    def generate(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool = True,
        candidate_index: int = 0,
    ) -> TeacherPrediction:
        expression = self._extract_expression(example.prompt)
        left, symbol, right, result = self._evaluate(expression)
        logits = tuple(-abs(result - label) * 1.5 for label in self.labels)
        probabilities = softmax(logits)
        answer_index = self.labels.index(result) if result in self.labels else max(
            range(len(logits)), key=logits.__getitem__
        )
        reasoning = (
            f"Compute {left} {symbol} {right}. The result is {result}."
            if include_reasoning
            else ""
        )
        return TeacherPrediction(
            example_id=example.id,
            prompt=example.prompt,
            answer=str(result),
            reasoning=reasoning,
            logits=logits,
            confidence=probabilities[answer_index],
            candidate_index=candidate_index,
            metadata={"expression": expression},
        )

    @staticmethod
    def _extract_expression(prompt: str) -> str:
        return prompt.removeprefix("What is ").removesuffix("?").strip()

    @classmethod
    def _evaluate(cls, expression: str) -> tuple[int, str, int, int]:
        parsed = ast.parse(expression, mode="eval").body
        if (
            not isinstance(parsed, ast.BinOp)
            or type(parsed.op) not in cls._OPERATORS
            or not isinstance(parsed.left, ast.Constant)
            or not isinstance(parsed.right, ast.Constant)
            or not isinstance(parsed.left.value, int)
            or not isinstance(parsed.right.value, int)
        ):
            raise ValueError(f"unsupported arithmetic expression: {expression!r}")
        operation = cls._OPERATORS[type(parsed.op)]
        symbols = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
        result = operation(parsed.left.value, parsed.right.value)
        if not math.isfinite(result):
            raise ValueError("non-finite arithmetic result")
        return parsed.left.value, symbols[type(parsed.op)], parsed.right.value, int(result)
