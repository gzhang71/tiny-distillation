"""Runnable offline experiment across all requested distillation strategies."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence

import torch

from tiny_distillation.core import TrainingExample
from tiny_distillation.inference import (
    SpeculativeDecodingConfig,
    speculative_decode,
)
from tiny_distillation.pipeline import DistillationPipeline
from tiny_distillation.teachers import RuleBasedArithmeticTeacher
from tiny_distillation.student_training import (
    DistillationMode,
    StudentTrainer,
    TinyStudent,
    TrainerConfig,
    Vocabulary,
)


class _FixedAutoregressiveModel:
    def __init__(self, logits: Sequence[float]) -> None:
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def next_token_logits(self, token_ids: tuple[int, ...]) -> torch.Tensor:
        return self.logits

    def next_token_logits_batch(
        self,
        prefixes: Sequence[tuple[int, ...]],
    ) -> torch.Tensor:
        return self.logits.repeat(len(prefixes), 1)


def build_arithmetic_data(seed: int) -> tuple[list[TrainingExample], list[TrainingExample]]:
    examples: list[TrainingExample] = []
    for left in range(9):
        for right in range(9):
            result = left + right
            examples.append(
                TrainingExample(
                    id=f"{left}-{right}",
                    prompt=f"What is {left} + {right}?",
                    label=result,
                    reference_answer=str(result),
                    metadata={
                        "reference_reasoning": (
                            f"Compute {left} + {right}. The result is {result}."
                        )
                    },
                )
            )
    random.Random(seed).shuffle(examples)
    return examples[:65], examples[65:]


def run_experiment(
    modes: Sequence[DistillationMode],
    *,
    epochs: int,
    seed: int,
) -> dict[str, object]:
    train_examples, evaluation_examples = build_arithmetic_data(seed)
    pipeline = DistillationPipeline(RuleBasedArithmeticTeacher(range(17)))
    artifacts = pipeline.prepare(train_examples)
    vocabulary = Vocabulary.from_labels(artifacts.labels)
    mode_results: dict[str, object] = {}

    for mode in modes:
        torch.manual_seed(seed)
        student = TinyStudent(
            len(vocabulary),
            num_labels=17,
            embedding_dim=32,
            hidden_dim=48,
            pad_id=vocabulary.pad_id,
        )
        trainer = StudentTrainer(
            student,
            vocabulary,
            TrainerConfig(
                mode=mode,
                epochs=epochs,
                batch_size=16,
                seed=seed,
            ),
        )
        history = pipeline.train(trainer, artifacts)
        report = pipeline.evaluate(
            trainer,
            evaluation_examples,
            include_reasoning_metrics=True,
        )
        mode_results[mode.value] = {
            "initial_loss": round(history.losses[0], 4),
            "final_loss": round(history.losses[-1], 4),
            "accuracy": round(report.accuracy, 4),
            "nll": round(report.negative_log_likelihood, 4),
            "ece": round(report.expected_calibration_error, 4),
            "reasoning_token_f1": round(report.reasoning_token_f1 or 0.0, 4),
        }

    draft = _FixedAutoregressiveModel([2.2, 1.0, 0.1, -0.5])
    target = _FixedAutoregressiveModel([2.0, 1.1, 0.2, -0.3])
    speculative = speculative_decode(
        (),
        draft,
        target,
        SpeculativeDecodingConfig(
            draft_tokens=4,
            max_new_tokens=24,
            seed=seed,
        ),
    )
    return {
        "training_examples": len(artifacts.labels),
        "calibrated_temperature": round(artifacts.fitted_temperature, 4),
        "modes": mode_results,
        "speculative_decoding": {
            "tokens": len(speculative.token_ids),
            "acceptance_rate": round(speculative.acceptance_rate, 4),
            "target_calls": speculative.target_calls,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in DistillationMode] + ["all"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    modes = (
        list(DistillationMode)
        if args.mode == "all"
        else [DistillationMode(args.mode)]
    )
    print(json.dumps(run_experiment(modes, epochs=args.epochs, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
