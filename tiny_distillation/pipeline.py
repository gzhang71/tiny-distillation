"""End-to-end orchestration while keeping each stage independently replaceable."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from tiny_distillation.core import (
    CalibratedLabel,
    ScoredPrediction,
    TeacherPrediction,
    TrainingExample,
    TrainingHistory,
)
from tiny_distillation.evaluation import EvaluationReport, evaluate_classification
from tiny_distillation.preparation import (
    CompositeScorer,
    LabelCalibrator,
    ReasoningGenerationConfig,
    generate_reasoning,
)
from tiny_distillation.teachers import Teacher
from tiny_distillation.training import StudentTrainer


@dataclass(frozen=True)
class PipelineArtifacts:
    predictions: tuple[TeacherPrediction, ...]
    scored_predictions: tuple[ScoredPrediction, ...]
    labels: tuple[CalibratedLabel, ...]
    fitted_temperature: float


class DistillationPipeline:
    def __init__(
        self,
        teacher: Teacher,
        *,
        scorer: CompositeScorer | None = None,
        calibrator: LabelCalibrator | None = None,
        generation_config: ReasoningGenerationConfig | None = None,
    ) -> None:
        self.teacher = teacher
        self.scorer = scorer or CompositeScorer()
        self.calibrator = calibrator or LabelCalibrator()
        self.generation_config = generation_config or ReasoningGenerationConfig()

    def prepare(self, examples: Sequence[TrainingExample]) -> PipelineArtifacts:
        predictions = generate_reasoning(
            examples,
            self.teacher,
            self.generation_config,
        )
        all_scores = self.scorer.score(predictions, examples)
        best_scores = self.scorer.best_per_example(all_scores)
        labels = self.calibrator.fit_transform(best_scores, examples)
        return PipelineArtifacts(
            predictions=tuple(predictions),
            scored_predictions=tuple(best_scores),
            labels=tuple(labels),
            fitted_temperature=self.calibrator.temperature,
        )

    @staticmethod
    def train(
        trainer: StudentTrainer,
        artifacts: PipelineArtifacts,
    ) -> TrainingHistory:
        return trainer.train(artifacts.labels)

    @staticmethod
    def evaluate(
        trainer: StudentTrainer,
        examples: Sequence[TrainingExample],
        *,
        include_reasoning_metrics: bool = False,
    ) -> EvaluationReport:
        labeled = [example for example in examples if example.label is not None]
        if not labeled:
            raise ValueError("evaluation requires examples with labels")
        logits, generated_reasoning = trainer.predict(
            [example.prompt for example in labeled]
        )
        reference_reasoning = None
        if include_reasoning_metrics:
            reference_reasoning = [
                str(example.metadata.get("reference_reasoning", ""))
                for example in labeled
            ]
        return evaluate_classification(
            logits,
            [int(example.label) for example in labeled],
            generated_reasoning=(
                generated_reasoning if include_reasoning_metrics else None
            ),
            reference_reasoning=reference_reasoning,
        )
