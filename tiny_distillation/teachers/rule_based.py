"""Deterministic teacher used by the demo and offline tests."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Sequence

from tiny_distillation.core.types import TrainingExample
from tiny_distillation.teachers.base import Teacher, TeacherResponse


class RuleBasedArithmeticTeacher(Teacher):
    provider = "rule-based"
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
    }

    def __init__(self, labels: Sequence[int]) -> None:
        self.numeric_labels = tuple(labels)
        super().__init__(model="arithmetic", labels=labels)

    def _request(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool,
        candidate_index: int,
    ) -> TeacherResponse:
        expression = self._extract_expression(example.prompt)
        left, symbol, right, result = self._evaluate(expression)
        return TeacherResponse(
            answer=str(result),
            reasoning=(
                f"Compute {left} {symbol} {right}. The result is {result}."
                if include_reasoning
                else ""
            ),
            confidence=1.0,
            metadata={"expression": expression, "result": result},
        )

    def project_logits(
        self,
        response: TeacherResponse,
        example: TrainingExample,
    ) -> tuple[float, ...]:
        result = int(response.metadata["result"])
        return tuple(-abs(result - label) * 1.5 for label in self.numeric_labels)

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

