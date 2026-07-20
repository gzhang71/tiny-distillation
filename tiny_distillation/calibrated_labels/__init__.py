"""Composable teacher-label calibration and target construction."""

from tiny_distillation.calibrated_labels.base import (
    CalibrationStrategy,
    LabelBuilder,
    LabelFilter,
    LabelTargets,
    WeightingStrategy,
)
from tiny_distillation.calibrated_labels.calibration import (
    IdentityCalibration,
    TemperatureCalibration,
)
from tiny_distillation.calibrated_labels.calibrator import (
    CalibrationConfig,
    CalibrationMethod,
    LabelBuildingMethod,
    LabelCalibrator,
    WeightingMethod,
)
from tiny_distillation.calibrated_labels.filtering import (
    AcceptedLabelFilter,
    CompositeLabelFilter,
    QualityLabelFilter,
)
from tiny_distillation.calibrated_labels.label_builders import (
    GroundTruthBlendLabelBuilder,
    TeacherLabelBuilder,
)
from tiny_distillation.calibrated_labels.weighting import (
    ConfidenceWeighting,
    EntropyWeighting,
    MarginWeighting,
    ScoreWeighting,
)

__all__ = [
    "AcceptedLabelFilter",
    "CalibrationConfig",
    "CalibrationMethod",
    "CalibrationStrategy",
    "CompositeLabelFilter",
    "ConfidenceWeighting",
    "EntropyWeighting",
    "GroundTruthBlendLabelBuilder",
    "IdentityCalibration",
    "LabelBuilder",
    "LabelBuildingMethod",
    "LabelCalibrator",
    "LabelFilter",
    "LabelTargets",
    "MarginWeighting",
    "QualityLabelFilter",
    "ScoreWeighting",
    "TeacherLabelBuilder",
    "TemperatureCalibration",
    "WeightingMethod",
    "WeightingStrategy",
]
