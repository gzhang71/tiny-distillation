"""Evaluation orchestration and backward-compatible report fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from tiny_distillation.evaluation.base import EvaluationContext, EvaluationMetric
from tiny_distillation.evaluation.classification import (
    AccuracyMetric,
    BrierScoreMetric,
    ExpectedCalibrationErrorMetric,
    MacroF1Metric,
    MacroPrecisionMetric,
    MacroRecallMetric,
    MaximumCalibrationErrorMetric,
    MeanConfidenceMetric,
    NegativeLogLikelihoodMetric,
    PredictiveEntropyMetric,
    TopKAccuracyMetric,
)
from tiny_distillation.evaluation.reasoning import (
    ReasoningExactMatchMetric,
    ReasoningTokenF1Metric,
    ReasoningTokenPrecisionMetric,
    ReasoningTokenRecallMetric,
)


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    negative_log_likelihood: float
    brier_score: float
    expected_calibration_error: float
    reasoning_exact_match: float | None = None
    reasoning_token_f1: float | None = None
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    top_k_accuracy: float = 0.0
    mean_confidence: float = 0.0
    predictive_entropy: float = 0.0
    maximum_calibration_error: float = 0.0
    reasoning_token_precision: float | None = None
    reasoning_token_recall: float | None = None
    metric_values: Mapping[str, float] = field(default_factory=dict)

    def __getitem__(self, metric_name: str) -> float:
        return self.metric_values[metric_name]


def evaluate_classification(
    logits: Tensor | Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    num_bins: int = 10,
    top_k: int = 3,
    generated_reasoning: Sequence[str] | None = None,
    reference_reasoning: Sequence[str] | None = None,
    additional_metrics: Sequence[EvaluationMetric] = (),
) -> EvaluationReport:
    context = _build_context(
        logits,
        targets,
        generated_reasoning,
        reference_reasoning,
    )
    metrics: list[EvaluationMetric] = [
        AccuracyMetric(),
        NegativeLogLikelihoodMetric(),
        BrierScoreMetric(),
        ExpectedCalibrationErrorMetric(num_bins),
        MacroPrecisionMetric(),
        MacroRecallMetric(),
        MacroF1Metric(),
        TopKAccuracyMetric(top_k),
        MeanConfidenceMetric(),
        PredictiveEntropyMetric(),
        MaximumCalibrationErrorMetric(num_bins),
    ]
    if context.generated_reasoning is not None:
        metrics.extend(
            [
                ReasoningExactMatchMetric(),
                ReasoningTokenPrecisionMetric(),
                ReasoningTokenRecallMetric(),
                ReasoningTokenF1Metric(),
            ]
        )
    metrics.extend(additional_metrics)

    values: dict[str, float] = {}
    for metric in metrics:
        if metric.name in values:
            raise ValueError(f"duplicate evaluation metric name: {metric.name}")
        values[metric.name] = metric.compute(context)

    return EvaluationReport(
        accuracy=values["accuracy"],
        negative_log_likelihood=values["negative_log_likelihood"],
        brier_score=values["brier_score"],
        expected_calibration_error=values["expected_calibration_error"],
        macro_precision=values["macro_precision"],
        macro_recall=values["macro_recall"],
        macro_f1=values["macro_f1"],
        top_k_accuracy=values["top_k_accuracy"],
        mean_confidence=values["mean_confidence"],
        predictive_entropy=values["predictive_entropy"],
        maximum_calibration_error=values["maximum_calibration_error"],
        reasoning_exact_match=values.get("reasoning_exact_match"),
        reasoning_token_precision=values.get("reasoning_token_precision"),
        reasoning_token_recall=values.get("reasoning_token_recall"),
        reasoning_token_f1=values.get("reasoning_token_f1"),
        metric_values=values,
    )


def _build_context(
    logits: Tensor | Sequence[Sequence[float]],
    targets: Sequence[int],
    generated_reasoning: Sequence[str] | None,
    reference_reasoning: Sequence[str] | None,
) -> EvaluationContext:
    scores = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(targets, dtype=torch.long)
    if scores.ndim != 2 or len(scores) != len(labels) or len(labels) == 0:
        raise ValueError("logits must be [examples, labels] and match non-empty targets")
    if bool(labels.lt(0).any()) or bool(labels.ge(scores.shape[1]).any()):
        raise ValueError("targets must be valid class indices")
    if generated_reasoning is not None or reference_reasoning is not None:
        if (
            generated_reasoning is None
            or reference_reasoning is None
            or len(generated_reasoning) != len(reference_reasoning)
            or len(generated_reasoning) != len(labels)
        ):
            raise ValueError(
                "generated and reference reasoning must match the number of targets"
            )

    probabilities = scores.softmax(dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    return EvaluationContext(
        logits=scores,
        targets=labels,
        probabilities=probabilities,
        predictions=predictions,
        confidence=confidence,
        correct=predictions.eq(labels),
        generated_reasoning=(
            tuple(generated_reasoning) if generated_reasoning is not None else None
        ),
        reference_reasoning=(
            tuple(reference_reasoning) if reference_reasoning is not None else None
        ),
    )
