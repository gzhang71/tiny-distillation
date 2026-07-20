"""Shared evaluation context and metric contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class EvaluationContext:
    logits: Tensor
    targets: Tensor
    probabilities: Tensor
    predictions: Tensor
    confidence: Tensor
    correct: Tensor
    generated_reasoning: tuple[str, ...] | None = None
    reference_reasoning: tuple[str, ...] | None = None


class EvaluationMetric(ABC):
    """Computes one scalar from a validated evaluation context."""

    name: str

    @abstractmethod
    def compute(self, context: EvaluationContext) -> float:
        ...

