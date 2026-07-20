import unittest

from tiny_distillation.core import TrainingExample
from tiny_distillation.generate_reasoning import (
    CritiqueRevisionStrategy,
    ReasoningGenerationConfig,
    ReasoningStrategy,
    ReasoningStrategyName,
    generate_reasoning,
)
from tiny_distillation.teachers import Teacher, TeacherResponse


class RecordingTeacher(Teacher):
    provider = "recording"

    def __init__(self) -> None:
        super().__init__("recording", ["no", "yes"])
        self.calls: list[tuple[str, bool, int]] = []

    def _request(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool,
        candidate_index: int,
    ) -> TeacherResponse:
        instruction = str(example.metadata.get("generation_instruction", ""))
        self.calls.append((instruction, include_reasoning, candidate_index))
        return TeacherResponse(
            answer="yes",
            reasoning=f"Reasoning for candidate {candidate_index}" if include_reasoning else "",
            confidence=0.9,
        )


class CustomStrategy(ReasoningStrategy):
    name = "custom"

    def build_instruction(
        self,
        example: TrainingExample,
        candidate_index: int,
    ) -> str:
        return "Use the custom reasoning procedure."


class ReasoningStrategiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.example = TrainingExample(id="one", prompt="Is one positive?")

    def test_every_builtin_strategy_is_configurable(self) -> None:
        for strategy_name in ReasoningStrategyName:
            with self.subTest(strategy=strategy_name):
                teacher = RecordingTeacher()
                predictions = generate_reasoning(
                    [self.example],
                    teacher,
                    ReasoningGenerationConfig(strategy=strategy_name),
                )

                self.assertEqual(len(predictions), 1)
                self.assertEqual(
                    predictions[0].metadata["reasoning_strategy"],
                    strategy_name.value,
                )

    def test_direct_strategy_disables_reasoning_by_default(self) -> None:
        teacher = RecordingTeacher()

        prediction = generate_reasoning(
            [self.example],
            teacher,
            ReasoningGenerationConfig(strategy="direct"),
        )[0]

        self.assertEqual(prediction.reasoning, "")
        self.assertFalse(teacher.calls[0][1])

    def test_self_consistency_generates_independent_candidates(self) -> None:
        teacher = RecordingTeacher()

        predictions = generate_reasoning(
            [self.example],
            teacher,
            ReasoningGenerationConfig(
                strategy="self_consistency",
                candidates_per_example=3,
            ),
        )

        self.assertEqual(len(predictions), 3)
        self.assertEqual([item.candidate_index for item in predictions], [0, 1, 2])
        self.assertEqual(len({call[0] for call in teacher.calls}), 3)

    def test_critique_revision_uses_two_teacher_passes(self) -> None:
        teacher = RecordingTeacher()

        prediction = generate_reasoning(
            [self.example],
            teacher,
            ReasoningGenerationConfig(strategy=CritiqueRevisionStrategy()),
        )[0]

        self.assertEqual(len(teacher.calls), 2)
        self.assertIn("Draft answer", teacher.calls[1][0])
        self.assertEqual(prediction.metadata["draft_answer"], "yes")
        self.assertIn("draft_reasoning", prediction.metadata)

    def test_custom_strategy_and_instruction(self) -> None:
        teacher = RecordingTeacher()

        prediction = generate_reasoning(
            [self.example],
            teacher,
            ReasoningGenerationConfig(
                strategy=CustomStrategy(),
                custom_instruction="Keep it under three sentences.",
            ),
        )[0]

        self.assertEqual(prediction.metadata["reasoning_strategy"], "custom")
        self.assertIn("under three sentences", teacher.calls[0][0])

    def test_candidate_deduplication(self) -> None:
        teacher = RecordingTeacher()

        predictions = generate_reasoning(
            [self.example],
            teacher,
            ReasoningGenerationConfig(
                strategy="direct",
                candidates_per_example=3,
                deduplicate_candidates=True,
            ),
        )

        self.assertEqual(len(predictions), 1)

    def test_unknown_strategy_is_rejected_by_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown reasoning strategy"):
            ReasoningGenerationConfig(strategy="unknown")


if __name__ == "__main__":
    unittest.main()
