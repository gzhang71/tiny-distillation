import unittest

from tiny_distillation.core import TeacherPrediction, TrainingExample
from tiny_distillation.score import (
    CompositeScorer,
    ConfidenceScorer,
    ConsensusScorer,
    ConsensusScoringConfig,
    ExactAnswerScorer,
    ReasoningQualityScorer,
    RewardScorer,
    ScoringStrategy,
)


def prediction(
    example_id: str,
    answer: str,
    *,
    reasoning: str = "The evidence supports the answer.",
    confidence: float = 0.8,
    candidate_index: int = 0,
) -> TeacherPrediction:
    return TeacherPrediction(
        example_id=example_id,
        prompt="Choose yes or no.",
        answer=answer,
        reasoning=reasoning,
        logits=(0.0, 1.0),
        confidence=confidence,
        candidate_index=candidate_index,
    )


class ScoringStrategiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.example = TrainingExample(
            id="decision",
            prompt="Choose yes or no.",
            reference_answer="yes",
        )

    def test_every_scorer_uses_the_shared_base(self) -> None:
        scorers = (
            CompositeScorer(),
            ExactAnswerScorer(),
            ConfidenceScorer(),
            ReasoningQualityScorer(),
            RewardScorer(lambda prediction, example: 1.0),
            ConsensusScorer(),
        )
        self.assertTrue(all(isinstance(scorer, ScoringStrategy) for scorer in scorers))

    def test_exact_answer_scorer_ignores_confidence(self) -> None:
        predictions = [
            prediction("decision", "yes", confidence=0.1),
            prediction("decision", "no", confidence=0.99, candidate_index=1),
        ]

        scored = ExactAnswerScorer().score(predictions, [self.example])

        self.assertEqual([item.total_score for item in scored], [1.0, 0.0])
        self.assertEqual([item.accepted for item in scored], [True, False])

    def test_confidence_scorer_ignores_answer_correctness(self) -> None:
        scored = ConfidenceScorer(acceptance_threshold=0.9).score(
            [prediction("decision", "no", confidence=0.95)],
            [self.example],
        )[0]

        self.assertEqual(scored.total_score, 0.95)
        self.assertTrue(scored.accepted)

    def test_reasoning_quality_scorer_rejects_empty_trace(self) -> None:
        scored = ReasoningQualityScorer().score(
            [prediction("decision", "yes", reasoning="")],
            [self.example],
        )[0]

        self.assertEqual(scored.total_score, 0.0)
        self.assertFalse(scored.accepted)

    def test_reward_scorer_uses_external_verifier_as_total(self) -> None:
        scored = RewardScorer(
            lambda prediction, example: 0.25,
            acceptance_threshold=0.5,
        ).score(
            [prediction("decision", "yes")],
            [self.example],
        )[0]

        self.assertEqual(scored.total_score, 0.25)
        self.assertEqual(scored.diagnostics["external_reward"], 0.25)
        self.assertFalse(scored.accepted)

    def test_consensus_scorer_rewards_majority_answer(self) -> None:
        predictions = [
            prediction("decision", "yes", candidate_index=0),
            prediction("decision", "yes", candidate_index=1),
            prediction("decision", "no", candidate_index=2),
        ]
        scorer = ConsensusScorer(
            base_scorer=ConfidenceScorer(acceptance_threshold=0.0),
            config=ConsensusScoringConfig(
                consensus_weight=1.0,
                acceptance_threshold=0.5,
            ),
        )

        scored = scorer.score(predictions, [self.example])

        self.assertEqual(
            [round(item.total_score, 3) for item in scored],
            [0.667, 0.667, 0.333],
        )
        self.assertEqual([item.accepted for item in scored], [True, True, False])

    def test_consensus_requires_multiple_candidates(self) -> None:
        scored = ConsensusScorer(
            config=ConsensusScoringConfig(consensus_weight=1.0)
        ).score(
            [prediction("decision", "yes")],
            [self.example],
        )[0]

        self.assertEqual(scored.total_score, 0.0)
        self.assertEqual(scored.diagnostics["candidate_count"], 1.0)


if __name__ == "__main__":
    unittest.main()

