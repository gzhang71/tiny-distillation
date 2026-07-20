import unittest

from tiny_distillation.evaluation import (
    AccuracyMetric,
    BrierScoreMetric,
    EvaluationContext,
    EvaluationMetric,
    ExpectedCalibrationErrorMetric,
    MacroF1Metric,
    MacroPrecisionMetric,
    MacroRecallMetric,
    MaximumCalibrationErrorMetric,
    MeanConfidenceMetric,
    NegativeLogLikelihoodMetric,
    PredictiveEntropyMetric,
    ReasoningExactMatchMetric,
    ReasoningTokenF1Metric,
    ReasoningTokenPrecisionMetric,
    ReasoningTokenRecallMetric,
    TopKAccuracyMetric,
    evaluate_classification,
)


class ConstantMetric(EvaluationMetric):
    name = "constant"

    def compute(self, context: EvaluationContext) -> float:
        return 0.42


class EvaluationMetricsTest(unittest.TestCase):
    def test_all_metrics_share_the_abstract_base(self) -> None:
        metrics = (
            AccuracyMetric(),
            NegativeLogLikelihoodMetric(),
            BrierScoreMetric(),
            ExpectedCalibrationErrorMetric(),
            MaximumCalibrationErrorMetric(),
            MacroPrecisionMetric(),
            MacroRecallMetric(),
            MacroF1Metric(),
            TopKAccuracyMetric(),
            MeanConfidenceMetric(),
            PredictiveEntropyMetric(),
            ReasoningExactMatchMetric(),
            ReasoningTokenPrecisionMetric(),
            ReasoningTokenRecallMetric(),
            ReasoningTokenF1Metric(),
        )
        self.assertTrue(all(isinstance(metric, EvaluationMetric) for metric in metrics))

    def test_perfect_predictions_have_perfect_macro_and_top_k_metrics(self) -> None:
        report = evaluate_classification(
            [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            [0, 1, 2],
            top_k=2,
        )

        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.macro_precision, 1.0)
        self.assertEqual(report.macro_recall, 1.0)
        self.assertEqual(report.macro_f1, 1.0)
        self.assertEqual(report.top_k_accuracy, 1.0)
        self.assertLess(report.maximum_calibration_error, 0.01)

    def test_entropy_and_confidence_reflect_uncertainty(self) -> None:
        confident = evaluate_classification([[8.0, 0.0]], [0])
        uncertain = evaluate_classification([[0.0, 0.0]], [0])

        self.assertGreater(confident.mean_confidence, uncertain.mean_confidence)
        self.assertLess(confident.predictive_entropy, uncertain.predictive_entropy)

    def test_reasoning_precision_recall_and_f1(self) -> None:
        report = evaluate_classification(
            [[2.0, 0.0]],
            [0],
            generated_reasoning=["alpha beta"],
            reference_reasoning=["alpha beta gamma"],
        )

        self.assertEqual(report.reasoning_exact_match, 0.0)
        self.assertEqual(report.reasoning_token_precision, 1.0)
        self.assertAlmostEqual(report.reasoning_token_recall or 0.0, 2 / 3)
        self.assertAlmostEqual(report.reasoning_token_f1 or 0.0, 0.8)

    def test_custom_metric_is_available_by_name(self) -> None:
        report = evaluate_classification(
            [[2.0, 0.0]],
            [0],
            additional_metrics=[ConstantMetric()],
        )

        self.assertEqual(report["constant"], 0.42)
        self.assertEqual(report.metric_values["constant"], 0.42)

    def test_duplicate_metric_name_is_rejected(self) -> None:
        class DuplicateAccuracy(EvaluationMetric):
            name = "accuracy"

            def compute(self, context: EvaluationContext) -> float:
                return 0.0

        with self.assertRaisesRegex(ValueError, "duplicate evaluation metric"):
            evaluate_classification(
                [[2.0, 0.0]],
                [0],
                additional_metrics=[DuplicateAccuracy()],
            )


if __name__ == "__main__":
    unittest.main()

