"""Student quality and calibration metrics."""

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
from tiny_distillation.evaluation.metrics import (
    EvaluationReport,
    evaluate_classification,
)
from tiny_distillation.evaluation.reasoning import (
    ReasoningExactMatchMetric,
    ReasoningTokenF1Metric,
    ReasoningTokenPrecisionMetric,
    ReasoningTokenRecallMetric,
)

__all__ = [
    "AccuracyMetric",
    "BrierScoreMetric",
    "EvaluationContext",
    "EvaluationMetric",
    "EvaluationReport",
    "ExpectedCalibrationErrorMetric",
    "MacroF1Metric",
    "MacroPrecisionMetric",
    "MacroRecallMetric",
    "MaximumCalibrationErrorMetric",
    "MeanConfidenceMetric",
    "NegativeLogLikelihoodMetric",
    "PredictiveEntropyMetric",
    "ReasoningExactMatchMetric",
    "ReasoningTokenF1Metric",
    "ReasoningTokenPrecisionMetric",
    "ReasoningTokenRecallMetric",
    "TopKAccuracyMetric",
    "evaluate_classification",
]
