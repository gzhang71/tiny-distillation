"""Composable conversion from scored predictions to trainable targets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from tiny_distillation.calibrated_labels.base import (
    CalibrationStrategy,
    LabelBuilder,
    LabelFilter,
    WeightingStrategy,
)
from tiny_distillation.calibrated_labels.calibration import (
    IdentityCalibration,
    TemperatureCalibration,
)
from tiny_distillation.calibrated_labels.filtering import (
    AcceptedLabelFilter,
    CompositeLabelFilter,
    QualityLabelFilter,
    normalized_entropy,
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
from tiny_distillation.core.types import (
    CalibratedLabel,
    ScoredPrediction,
    TrainingExample,
)


class CalibrationMethod(str, Enum):
    IDENTITY = "identity"
    TEMPERATURE = "temperature"


class LabelBuildingMethod(str, Enum):
    TEACHER = "teacher"
    GROUND_TRUTH_BLEND = "ground_truth_blend"


class WeightingMethod(str, Enum):
    SCORE = "score"
    CONFIDENCE = "confidence"
    ENTROPY = "entropy"
    MARGIN = "margin"


@dataclass(frozen=True)
class CalibrationConfig:
    calibration_method: CalibrationMethod | str = CalibrationMethod.TEMPERATURE
    temperature: float = 1.0
    fit_temperature: bool = True
    temperature_min: float = 0.25
    temperature_max: float = 5.0
    temperature_steps: int = 80
    label_building_method: LabelBuildingMethod | str = LabelBuildingMethod.TEACHER
    ground_truth_weight: float = 0.5
    label_smoothing: float = 0.0
    top_k: int | None = None
    accepted_only: bool = True
    minimum_confidence: float = 0.0
    maximum_entropy: float = 1.0
    minimum_margin: float = 0.0
    weighting_method: WeightingMethod | str = WeightingMethod.SCORE
    minimum_weight: float = 0.10

    def __post_init__(self) -> None:
        _enum_value(CalibrationMethod, self.calibration_method, "calibration_method")
        _enum_value(
            LabelBuildingMethod,
            self.label_building_method,
            "label_building_method",
        )
        _enum_value(WeightingMethod, self.weighting_method, "weighting_method")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.temperature_min <= 0 or self.temperature_max < self.temperature_min:
            raise ValueError("invalid temperature search range")
        if self.temperature_steps < 2:
            raise ValueError("temperature_steps must be at least 2")
        if not 0 <= self.ground_truth_weight <= 1:
            raise ValueError("ground_truth_weight must be in [0, 1]")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if not 0 <= self.maximum_entropy <= 1:
            raise ValueError("maximum_entropy must be in [0, 1]")
        if not 0 <= self.minimum_margin <= 1:
            raise ValueError("minimum_margin must be in [0, 1]")
        if not 0 <= self.minimum_weight <= 1:
            raise ValueError("minimum_weight must be in [0, 1]")


class LabelCalibrator:
    """Composes calibration, target building, filtering, and weighting."""

    def __init__(
        self,
        config: CalibrationConfig | None = None,
        *,
        calibration_strategy: CalibrationStrategy | None = None,
        label_builder: LabelBuilder | None = None,
        label_filter: LabelFilter | None = None,
        weighting_strategy: WeightingStrategy | None = None,
    ) -> None:
        self.config = config or CalibrationConfig()
        self.calibration_strategy = (
            calibration_strategy or self._build_calibration_strategy()
        )
        self.label_builder = label_builder or self._build_label_builder()
        self.label_filter = label_filter or self._build_label_filter()
        self.weighting_strategy = (
            weighting_strategy or self._build_weighting_strategy()
        )

    @property
    def temperature(self) -> float:
        return self.calibration_strategy.temperature

    def fit(
        self,
        scored: Iterable[ScoredPrediction],
        examples: Iterable[TrainingExample],
    ) -> float:
        self.calibration_strategy.fit(scored, examples)
        return self.temperature

    def transform(
        self,
        scored: Iterable[ScoredPrediction],
        examples: Iterable[TrainingExample] | None = None,
    ) -> list[CalibratedLabel]:
        examples_by_id = {
            example.id: example for example in (examples if examples is not None else ())
        }
        labels: list[CalibratedLabel] = []
        for item in scored:
            probabilities = self.calibration_strategy.calibrate(
                item.prediction.logits
            )
            if not self.label_filter.keep(item, probabilities):
                continue
            example = examples_by_id.get(item.prediction.example_id)
            targets = self.label_builder.build(item, probabilities, example)
            ordered_probabilities = sorted(probabilities, reverse=True)
            confidence = ordered_probabilities[0]
            margin = (
                confidence - ordered_probabilities[1]
                if len(ordered_probabilities) > 1
                else confidence
            )
            labels.append(
                CalibratedLabel(
                    example_id=item.prediction.example_id,
                    prompt=item.prediction.prompt,
                    hard_label=targets.hard_label,
                    soft_labels=targets.soft_labels,
                    answer=item.prediction.answer,
                    reasoning=item.prediction.reasoning,
                    weight=self.weighting_strategy.weight(item, probabilities),
                    source_score=item.total_score,
                    metadata={
                        "calibration_strategy": self.calibration_strategy.name,
                        "label_builder": self.label_builder.name,
                        "label_filter": self.label_filter.name,
                        "weighting_strategy": self.weighting_strategy.name,
                        "calibrated_confidence": confidence,
                        "normalized_entropy": normalized_entropy(probabilities),
                        "probability_margin": margin,
                        "temperature": self.temperature,
                    },
                )
            )
        return labels

    def fit_transform(
        self,
        scored: Iterable[ScoredPrediction],
        examples: Iterable[TrainingExample],
    ) -> list[CalibratedLabel]:
        scored_items = list(scored)
        example_items = list(examples)
        self.fit(scored_items, example_items)
        return self.transform(scored_items, example_items)

    def _build_calibration_strategy(self) -> CalibrationStrategy:
        method = _enum_value(
            CalibrationMethod,
            self.config.calibration_method,
            "calibration_method",
        )
        if method is CalibrationMethod.IDENTITY:
            return IdentityCalibration()
        return TemperatureCalibration(
            self.config.temperature,
            fit_temperature=self.config.fit_temperature,
            temperature_min=self.config.temperature_min,
            temperature_max=self.config.temperature_max,
            temperature_steps=self.config.temperature_steps,
        )

    def _build_label_builder(self) -> LabelBuilder:
        method = _enum_value(
            LabelBuildingMethod,
            self.config.label_building_method,
            "label_building_method",
        )
        options = {
            "label_smoothing": self.config.label_smoothing,
            "top_k": self.config.top_k,
        }
        if method is LabelBuildingMethod.GROUND_TRUTH_BLEND:
            return GroundTruthBlendLabelBuilder(
                ground_truth_weight=self.config.ground_truth_weight,
                **options,
            )
        return TeacherLabelBuilder(**options)

    def _build_label_filter(self) -> LabelFilter:
        return CompositeLabelFilter(
            [
                AcceptedLabelFilter(self.config.accepted_only),
                QualityLabelFilter(
                    minimum_confidence=self.config.minimum_confidence,
                    maximum_entropy=self.config.maximum_entropy,
                    minimum_margin=self.config.minimum_margin,
                ),
            ]
        )

    def _build_weighting_strategy(self) -> WeightingStrategy:
        method = _enum_value(
            WeightingMethod,
            self.config.weighting_method,
            "weighting_method",
        )
        strategy_types: dict[WeightingMethod, type[WeightingStrategy]] = {
            WeightingMethod.SCORE: ScoreWeighting,
            WeightingMethod.CONFIDENCE: ConfidenceWeighting,
            WeightingMethod.ENTROPY: EntropyWeighting,
            WeightingMethod.MARGIN: MarginWeighting,
        }
        return strategy_types[method](self.config.minimum_weight)


def _enum_value(
    enum_type: type[Enum],
    value: Enum | str,
    field_name: str,
) -> Enum:
    try:
        return enum_type(value)
    except ValueError as error:
        options = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {options}") from error
