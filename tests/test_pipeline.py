import math
import unittest

import torch

from tiny_distillation.core import TrainingExample
from tiny_distillation.evaluation import evaluate_classification
from tiny_distillation.pipeline import DistillationPipeline
from tiny_distillation.preparation import CalibrationConfig, LabelCalibrator
from tiny_distillation.teachers import RuleBasedArithmeticTeacher
from tiny_distillation.training import (
    DistillationMode,
    StudentTrainer,
    TinyStudent,
    TrainerConfig,
    Vocabulary,
)


def examples() -> list[TrainingExample]:
    return [
        TrainingExample(
            id=f"example-{value}",
            prompt=f"What is {value} + 1?",
            label=value + 1,
            reference_answer=str(value + 1),
        )
        for value in range(4)
    ]


class PipelineTest(unittest.TestCase):
    def test_prepare_generates_scored_calibrated_labels(self) -> None:
        items = examples()
        pipeline = DistillationPipeline(
            RuleBasedArithmeticTeacher(range(5)),
            calibrator=LabelCalibrator(
                CalibrationConfig(fit_temperature=True, temperature_steps=10)
            ),
        )
        artifacts = pipeline.prepare(items)

        self.assertEqual(len(artifacts.predictions), len(items))
        self.assertEqual(len(artifacts.labels), len(items))
        self.assertTrue(all(item.reasoning for item in artifacts.labels))
        self.assertTrue(
            all(abs(sum(item.soft_labels) - 1.0) < 1e-6 for item in artifacts.labels)
        )
        self.assertGreater(artifacts.fitted_temperature, 0)

    def test_all_training_modes_produce_finite_loss(self) -> None:
        pipeline = DistillationPipeline(RuleBasedArithmeticTeacher(range(5)))
        artifacts = pipeline.prepare(examples())
        vocabulary = Vocabulary.from_labels(artifacts.labels)

        for mode in DistillationMode:
            with self.subTest(mode=mode):
                torch.manual_seed(3)
                model = TinyStudent(
                    len(vocabulary),
                    5,
                    embedding_dim=8,
                    hidden_dim=12,
                    pad_id=vocabulary.pad_id,
                )
                trainer = StudentTrainer(
                    model,
                    vocabulary,
                    TrainerConfig(
                        mode=mode,
                        epochs=2,
                        batch_size=4,
                        learning_rate=1e-2,
                    ),
                )
                history = trainer.train(artifacts.labels)
                self.assertEqual(len(history.losses), 2)
                self.assertTrue(all(math.isfinite(loss) for loss in history.losses))

    def test_evaluation_metrics(self) -> None:
        report = evaluate_classification(
            [[8.0, 0.0], [0.0, 8.0]],
            [0, 1],
        )
        self.assertEqual(report.accuracy, 1.0)
        self.assertLess(report.negative_log_likelihood, 0.01)
        self.assertLess(report.brier_score, 0.01)


if __name__ == "__main__":
    unittest.main()
