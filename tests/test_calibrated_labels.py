import unittest

from tiny_distillation.calibrated_labels import (
    AcceptedLabelFilter,
    CalibrationConfig,
    CalibrationStrategy,
    ConfidenceWeighting,
    EntropyWeighting,
    GroundTruthBlendLabelBuilder,
    IdentityCalibration,
    LabelBuilder,
    LabelCalibrator,
    LabelFilter,
    MarginWeighting,
    QualityLabelFilter,
    ScoreWeighting,
    TeacherLabelBuilder,
    TemperatureCalibration,
    WeightingStrategy,
)
from tiny_distillation.core import (
    ScoredPrediction,
    TeacherPrediction,
    TrainingExample,
)


def scored_prediction(
    *,
    logits: tuple[float, ...] = (2.0, 1.0, 0.0),
    total_score: float = 0.8,
    accepted: bool = True,
) -> ScoredPrediction:
    prediction = TeacherPrediction(
        example_id="example-1",
        prompt="Pick a class",
        answer="A",
        reasoning="A has the strongest evidence.",
        logits=logits,
        confidence=0.8,
    )
    return ScoredPrediction(
        prediction=prediction,
        answer_score=1.0,
        reasoning_score=0.8,
        confidence_score=0.8,
        total_score=total_score,
        accepted=accepted,
    )


class CalibratedLabelsTest(unittest.TestCase):
    def test_all_components_share_their_abstract_contracts(self) -> None:
        self.assertIsInstance(IdentityCalibration(), CalibrationStrategy)
        self.assertIsInstance(TeacherLabelBuilder(), LabelBuilder)
        self.assertIsInstance(AcceptedLabelFilter(), LabelFilter)
        for strategy in (
            ScoreWeighting(),
            ConfidenceWeighting(),
            EntropyWeighting(),
            MarginWeighting(),
        ):
            self.assertIsInstance(strategy, WeightingStrategy)

    def test_temperature_calibration_fits_supervised_logits(self) -> None:
        strategy = TemperatureCalibration(temperature_steps=12)
        item = scored_prediction(logits=(4.0, 0.0))
        example = TrainingExample(id="example-1", prompt="Pick", label=0)

        strategy.fit([item], [example])

        self.assertLess(strategy.temperature, 1.0)
        self.assertGreater(strategy.calibrate(item.prediction.logits)[0], 0.98)

    def test_teacher_targets_support_top_k_and_smoothing(self) -> None:
        builder = TeacherLabelBuilder(top_k=1, label_smoothing=0.1)

        targets = builder.build(
            scored_prediction(),
            (0.7, 0.2, 0.1),
            None,
        )

        self.assertEqual(targets.hard_label, 0)
        self.assertAlmostEqual(sum(targets.soft_labels), 1.0)
        self.assertAlmostEqual(targets.soft_labels[0], 0.9333333333)
        self.assertAlmostEqual(targets.soft_labels[1], 0.0333333333)

    def test_ground_truth_blending_overrides_hard_target(self) -> None:
        builder = GroundTruthBlendLabelBuilder(ground_truth_weight=0.75)
        example = TrainingExample(id="example-1", prompt="Pick", label=1)

        targets = builder.build(
            scored_prediction(),
            (0.8, 0.1, 0.1),
            example,
        )

        self.assertEqual(targets.hard_label, 1)
        self.assertEqual(targets.soft_labels, (0.2, 0.775, 0.025))

    def test_quality_filter_uses_confidence_entropy_and_margin(self) -> None:
        label_filter = QualityLabelFilter(
            minimum_confidence=0.7,
            maximum_entropy=0.7,
            minimum_margin=0.5,
        )
        item = scored_prediction()

        self.assertTrue(label_filter.keep(item, (0.8, 0.1, 0.1)))
        self.assertFalse(label_filter.keep(item, (0.5, 0.3, 0.2)))

    def test_config_composes_filter_builder_and_weighting(self) -> None:
        calibrator = LabelCalibrator(
            CalibrationConfig(
                calibration_method="identity",
                label_building_method="ground_truth_blend",
                ground_truth_weight=1.0,
                weighting_method="margin",
                minimum_weight=0.0,
                minimum_confidence=0.6,
            )
        )
        example = TrainingExample(id="example-1", prompt="Pick", label=1)

        labels = calibrator.fit_transform([scored_prediction()], [example])

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].hard_label, 1)
        self.assertGreater(labels[0].weight, 0.4)
        self.assertEqual(labels[0].metadata["calibration_strategy"], "identity")
        self.assertEqual(labels[0].metadata["label_builder"], "ground_truth_blend")
        self.assertEqual(labels[0].metadata["weighting_strategy"], "margin")
        self.assertGreater(labels[0].metadata["calibrated_confidence"], 0.6)
        self.assertGreater(labels[0].metadata["probability_margin"], 0.4)

    def test_rejected_or_uncertain_predictions_are_filtered(self) -> None:
        rejected = LabelCalibrator().transform(
            [scored_prediction(accepted=False)]
        )
        uncertain = LabelCalibrator(
            CalibrationConfig(
                fit_temperature=False,
                minimum_confidence=0.8,
            )
        ).transform([scored_prediction(logits=(0.0, 0.0, 0.0))])

        self.assertEqual(rejected, [])
        self.assertEqual(uncertain, [])

    def test_invalid_strategy_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "calibration_method"):
            CalibrationConfig(calibration_method="unknown")


if __name__ == "__main__":
    unittest.main()
