"""Classification, confidence, and calibration metric implementations."""

from __future__ import annotations

import torch

from tiny_distillation.evaluation.base import EvaluationContext, EvaluationMetric


class AccuracyMetric(EvaluationMetric):
    name = "accuracy"

    def compute(self, context: EvaluationContext) -> float:
        return float(context.correct.float().mean())


class NegativeLogLikelihoodMetric(EvaluationMetric):
    name = "negative_log_likelihood"

    def compute(self, context: EvaluationContext) -> float:
        target_probability = context.probabilities.gather(
            1,
            context.targets.unsqueeze(1),
        ).squeeze(1)
        return float(-target_probability.clamp_min(1e-12).log().mean())


class BrierScoreMetric(EvaluationMetric):
    name = "brier_score"

    def compute(self, context: EvaluationContext) -> float:
        one_hot = torch.nn.functional.one_hot(
            context.targets,
            context.logits.shape[1],
        ).float()
        return float(((context.probabilities - one_hot) ** 2).sum(dim=-1).mean())


class TopKAccuracyMetric(EvaluationMetric):
    name = "top_k_accuracy"

    def __init__(self, k: int = 3) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    def compute(self, context: EvaluationContext) -> float:
        k = min(self.k, context.logits.shape[1])
        top_indices = context.probabilities.topk(k, dim=-1).indices
        correct = top_indices.eq(context.targets.unsqueeze(1)).any(dim=1)
        return float(correct.float().mean())


class MacroPrecisionMetric(EvaluationMetric):
    name = "macro_precision"

    def compute(self, context: EvaluationContext) -> float:
        return _macro_average(context, "precision")


class MacroRecallMetric(EvaluationMetric):
    name = "macro_recall"

    def compute(self, context: EvaluationContext) -> float:
        return _macro_average(context, "recall")


class MacroF1Metric(EvaluationMetric):
    name = "macro_f1"

    def compute(self, context: EvaluationContext) -> float:
        return _macro_average(context, "f1")


class MeanConfidenceMetric(EvaluationMetric):
    name = "mean_confidence"

    def compute(self, context: EvaluationContext) -> float:
        return float(context.confidence.mean())


class PredictiveEntropyMetric(EvaluationMetric):
    name = "predictive_entropy"

    def compute(self, context: EvaluationContext) -> float:
        probabilities = context.probabilities.clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        return float(entropy.mean())


class ExpectedCalibrationErrorMetric(EvaluationMetric):
    name = "expected_calibration_error"

    def __init__(self, num_bins: int = 10) -> None:
        if num_bins < 1:
            raise ValueError("num_bins must be positive")
        self.num_bins = num_bins

    def compute(self, context: EvaluationContext) -> float:
        return sum(
            proportion * gap
            for proportion, gap in _calibration_bins(context, self.num_bins)
        )


class MaximumCalibrationErrorMetric(EvaluationMetric):
    name = "maximum_calibration_error"

    def __init__(self, num_bins: int = 10) -> None:
        if num_bins < 1:
            raise ValueError("num_bins must be positive")
        self.num_bins = num_bins

    def compute(self, context: EvaluationContext) -> float:
        gaps = [gap for _, gap in _calibration_bins(context, self.num_bins)]
        return max(gaps, default=0.0)


def _macro_average(context: EvaluationContext, metric: str) -> float:
    label_ids = torch.unique(torch.cat((context.targets, context.predictions)))
    values: list[float] = []
    for label in label_ids:
        predicted = context.predictions.eq(label)
        actual = context.targets.eq(label)
        true_positive = float((predicted & actual).sum())
        false_positive = float((predicted & ~actual).sum())
        false_negative = float((~predicted & actual).sum())
        precision = true_positive / max(1.0, true_positive + false_positive)
        recall = true_positive / max(1.0, true_positive + false_negative)
        if metric == "precision":
            value = precision
        elif metric == "recall":
            value = recall
        else:
            value = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
        values.append(value)
    return sum(values) / max(1, len(values))


def _calibration_bins(
    context: EvaluationContext,
    num_bins: int,
) -> list[tuple[float, float]]:
    boundaries = torch.linspace(
        0,
        1,
        num_bins + 1,
        device=context.confidence.device,
    )
    bins: list[tuple[float, float]] = []
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = context.confidence.gt(lower) & context.confidence.le(upper)
        if bool(in_bin.any()):
            proportion = float(in_bin.float().mean())
            accuracy = float(context.correct[in_bin].float().mean())
            average_confidence = float(context.confidence[in_bin].mean())
            bins.append((proportion, abs(accuracy - average_confidence)))
    return bins
