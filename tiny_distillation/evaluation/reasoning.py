"""Metrics for generated reasoning traces."""

from __future__ import annotations

from collections import Counter

from tiny_distillation.evaluation.base import EvaluationContext, EvaluationMetric
from tiny_distillation.score import normalize_answer


class ReasoningExactMatchMetric(EvaluationMetric):
    name = "reasoning_exact_match"

    def compute(self, context: EvaluationContext) -> float:
        pairs = _reasoning_pairs(context)
        return sum(
            normalize_answer(generated) == normalize_answer(reference)
            for generated, reference in pairs
        ) / len(pairs)


class ReasoningTokenPrecisionMetric(EvaluationMetric):
    name = "reasoning_token_precision"

    def compute(self, context: EvaluationContext) -> float:
        return _average_token_metric(context, "precision")


class ReasoningTokenRecallMetric(EvaluationMetric):
    name = "reasoning_token_recall"

    def compute(self, context: EvaluationContext) -> float:
        return _average_token_metric(context, "recall")


class ReasoningTokenF1Metric(EvaluationMetric):
    name = "reasoning_token_f1"

    def compute(self, context: EvaluationContext) -> float:
        return _average_token_metric(context, "f1")


def _reasoning_pairs(context: EvaluationContext) -> list[tuple[str, str]]:
    if (
        context.generated_reasoning is None
        or context.reference_reasoning is None
        or len(context.generated_reasoning) == 0
    ):
        raise ValueError("reasoning metrics require non-empty generated and reference traces")
    return list(zip(context.generated_reasoning, context.reference_reasoning))


def _average_token_metric(context: EvaluationContext, metric: str) -> float:
    values = [
        _token_metrics(generated, reference)[metric]
        for generated, reference in _reasoning_pairs(context)
    ]
    return sum(values) / len(values)


def _token_metrics(generated: str, reference: str) -> dict[str, float]:
    generated_tokens = normalize_answer(generated).split()
    reference_tokens = normalize_answer(reference).split()
    if not generated_tokens or not reference_tokens:
        exact = float(generated_tokens == reference_tokens)
        return {"precision": exact, "recall": exact, "f1": exact}

    generated_counts = Counter(generated_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum(
        min(count, reference_counts.get(token, 0))
        for token, count in generated_counts.items()
    )
    precision = overlap / len(generated_tokens)
    recall = overlap / len(reference_tokens)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}

