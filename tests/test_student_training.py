import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from tiny_distillation.core import CalibratedLabel
from tiny_distillation.student_training import (
    CoTTokenCrossEntropyLoss,
    CombinedDistillationLoss,
    DistillationLoss,
    HardCrossEntropyLoss,
    LinearCoTCurriculum,
    LossWeights,
    SoftKLDistillationLoss,
    StudentModel,
    StudentTrainer,
    TinyStudent,
    TrainerConfig,
    TransformerStudent,
    Vocabulary,
)


def labels(count: int = 8) -> list[CalibratedLabel]:
    return [
        CalibratedLabel(
            example_id=f"example-{index}",
            prompt=f"Choose class for item {index}",
            hard_label=index % 2,
            soft_labels=(
                (0.9, 0.1) if index % 2 == 0 else (0.1, 0.9)
            ),
            answer=str(index % 2),
            reasoning=(
                "This is a deliberately long rationale with several useful tokens."
            ),
            weight=0.5 + index / (2 * count),
            source_score=0.8,
            metadata={"calibration_strategy": "temperature"},
        )
        for index in range(count)
    ]


def tiny_trainer(
    items: list[CalibratedLabel],
    config: TrainerConfig,
) -> StudentTrainer:
    vocabulary = Vocabulary.from_labels(items)
    model = TinyStudent(
        len(vocabulary),
        2,
        embedding_dim=8,
        hidden_dim=12,
        pad_id=vocabulary.pad_id,
    )
    return StudentTrainer(model, vocabulary, config)


class StudentTrainingTest(unittest.TestCase):
    def test_models_and_losses_share_abstract_contracts(self) -> None:
        items = labels(2)
        vocabulary = Vocabulary.from_labels(items)
        tiny = TinyStudent(len(vocabulary), 2, pad_id=vocabulary.pad_id)
        transformer = TransformerStudent(
            len(vocabulary),
            2,
            embedding_dim=8,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            pad_id=vocabulary.pad_id,
        )

        self.assertIsInstance(tiny, StudentModel)
        self.assertIsInstance(transformer, StudentModel)
        for strategy in (
            HardCrossEntropyLoss(vocabulary.pad_id),
            SoftKLDistillationLoss(vocabulary.pad_id),
            CoTTokenCrossEntropyLoss(vocabulary.pad_id),
            CombinedDistillationLoss(vocabulary.pad_id),
        ):
            self.assertIsInstance(strategy, DistillationLoss)

    def test_soft_kl_applies_temperature_to_both_distributions(self) -> None:
        probabilities = torch.tensor([[0.8, 0.2]])
        class_logits = probabilities.log()
        batch = {
            "hard_labels": torch.tensor([0]),
            "soft_labels": probabilities,
            "weights": torch.tensor([1.0]),
            "reasoning_target_ids": torch.tensor([[1]]),
            "reasoning_token_mask": torch.tensor([[True]]),
        }
        output = SoftKLDistillationLoss(pad_id=0).compute(
            class_logits,
            torch.tensor([[[0.0, 2.0]]]),
            batch,
            soft_temperature=2.0,
            weights=LossWeights(),
        )

        self.assertLess(abs(float(output.soft)), 1e-6)

    def test_curriculum_introduces_cot_weight_linearly(self) -> None:
        scheduler = LinearCoTCurriculum(warmup_epochs=4)
        base = LossWeights(hard=1.0, soft=0.5, cot=0.8)

        self.assertAlmostEqual(scheduler.weights(0, 8, base).cot, 0.2)
        self.assertAlmostEqual(scheduler.weights(3, 8, base).cot, 0.8)

    def test_training_controls_emit_validation_and_schedule_history(self) -> None:
        items = labels()
        trainer = tiny_trainer(
            items,
            TrainerConfig(
                epochs=3,
                batch_size=2,
                learning_rate=1e-2,
                warmup_epochs=1,
                validation_fraction=0.25,
                early_stopping_patience=1,
                early_stopping_min_delta=1e6,
                gradient_accumulation_steps=2,
                sampling_method="class_balanced",
                cot_warmup_epochs=2,
                teacher_forcing_ratio=1.0,
                final_teacher_forcing_ratio=0.5,
                max_reasoning_tokens=4,
            ),
        )

        history = trainer.train(items)

        self.assertEqual(len(history.losses), len(history.validation_losses))
        self.assertEqual(len(history.losses), len(history.learning_rates))
        self.assertEqual(history.teacher_forcing_ratios[0], 1.0)
        self.assertLessEqual(history.teacher_forcing_ratios[-1], 0.75)
        self.assertIsNotNone(history.best_epoch)
        self.assertTrue(history.stopped_early)
        self.assertTrue(all(math.isfinite(loss) for loss in history.losses))
        batch = trainer._collate(items[:2])
        self.assertLessEqual(batch["reasoning_target_ids"].shape[1], 4)

    def test_checkpoint_and_log_include_training_and_evaluation(self) -> None:
        items = labels(4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "student.pt"
            log_path = Path(directory) / "experiment.json"
            trainer = tiny_trainer(
                items,
                TrainerConfig(
                    epochs=1,
                    batch_size=2,
                    checkpoint_path=str(checkpoint_path),
                    experiment_log_path=str(log_path),
                ),
            )

            trainer.train(items)
            trainer.record_evaluation({"accuracy": 0.75})

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            log = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["evaluation"]["accuracy"], 0.75)
            self.assertEqual(log["evaluation"]["accuracy"], 0.75)
            self.assertEqual(len(checkpoint["label_metadata"]), len(items))
            self.assertIn("training_history", checkpoint)
            self.assertEqual(checkpoint["trainer_config"]["mode"], "combined")

    def test_transformer_student_trains_and_predicts(self) -> None:
        items = labels(4)
        vocabulary = Vocabulary.from_labels(items)
        model = TransformerStudent(
            len(vocabulary),
            2,
            embedding_dim=8,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            dropout=0.0,
            pad_id=vocabulary.pad_id,
        )
        trainer = StudentTrainer(
            model,
            vocabulary,
            TrainerConfig(epochs=1, batch_size=2, max_reasoning_tokens=3),
        )

        history = trainer.train(items)
        logits, reasoning = trainer.predict([items[0].prompt])

        self.assertEqual(len(history.losses), 1)
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertEqual(len(reasoning), 1)


if __name__ == "__main__":
    unittest.main()
