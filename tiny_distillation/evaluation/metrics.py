"""Classification, calibration, and reasoning metrics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor

from tiny_distillation.score import normalize_answer


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    negative_log_likelihood: float
    brier_score: float
    expected_calibration_error: float
    reasoning_exact_match: float | None = None
    reasoning_token_f1: float | None = None


def evaluate_classification(
    logits: Tensor | Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    num_bins: int = 10,
    generated_reasoning: Sequence[str] | None = None,
    reference_reasoning: Sequence[str] | None = None,
) -> EvaluationReport:
    scores = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(targets, dtype=torch.long)
    if scores.ndim != 2 or len(scores) != len(labels) or len(labels) == 0:
        raise ValueError("logits must be [examples, labels] and match non-empty targets")
    probabilities = scores.softmax(dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(labels)
    accuracy = float(correct.float().mean())
    target_probability = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    nll = float(-target_probability.clamp_min(1e-12).log().mean())
    one_hot = torch.nn.functional.one_hot(labels, scores.shape[1]).float()
    brier = float(((probabilities - one_hot) ** 2).sum(dim=-1).mean())
    ece = _expected_calibration_error(confidence, correct, num_bins)

    exact_match: float | None = None
    token_f1: float | None = None
    if generated_reasoning is not None or reference_reasoning is not None:
        if (
            generated_reasoning is None
            or reference_reasoning is None
            or len(generated_reasoning) != len(reference_reasoning)
        ):
            raise ValueError("generated and reference reasoning must have equal lengths")
        exact_match = sum(
            normalize_answer(generated) == normalize_answer(reference)
            for generated, reference in zip(generated_reasoning, reference_reasoning)
        ) / max(1, len(generated_reasoning))
        token_f1 = sum(
            _token_f1(generated, reference)
            for generated, reference in zip(generated_reasoning, reference_reasoning)
        ) / max(1, len(generated_reasoning))

    return EvaluationReport(
        accuracy=accuracy,
        negative_log_likelihood=nll,
        brier_score=brier,
        expected_calibration_error=ece,
        reasoning_exact_match=exact_match,
        reasoning_token_f1=token_f1,
    )


def _expected_calibration_error(
    confidence: Tensor,
    correct: Tensor,
    num_bins: int,
) -> float:
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    boundaries = torch.linspace(0, 1, num_bins + 1)
    result = torch.tensor(0.0)
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = confidence.gt(lower) & confidence.le(upper)
        if bool(in_bin.any()):
            proportion = in_bin.float().mean()
            accuracy = correct[in_bin].float().mean()
            average_confidence = confidence[in_bin].mean()
            result += proportion * (accuracy - average_confidence).abs()
    return float(result)


def _token_f1(generated: str, reference: str) -> float:
    generated_tokens = normalize_answer(generated).split()
    reference_tokens = normalize_answer(reference).split()
    if not generated_tokens or not reference_tokens:
        return float(generated_tokens == reference_tokens)
    generated_counts = {token: generated_tokens.count(token) for token in set(generated_tokens)}
    reference_counts = {token: reference_tokens.count(token) for token in set(reference_tokens)}
    overlap = sum(
        min(count, reference_counts.get(token, 0))
        for token, count in generated_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(generated_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
