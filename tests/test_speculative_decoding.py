import unittest
from collections.abc import Sequence

import torch

from tiny_distillation.inference import (
    SpeculativeDecodingConfig,
    speculative_decode,
)


class FixedModel:
    def __init__(self, logits: list[float]) -> None:
        self.logits = torch.tensor(logits)

    def next_token_logits(self, token_ids: tuple[int, ...]) -> torch.Tensor:
        return self.logits

    def next_token_logits_batch(
        self,
        prefixes: Sequence[tuple[int, ...]],
    ) -> torch.Tensor:
        return self.logits.repeat(len(prefixes), 1)


class SpeculativeDecodingTest(unittest.TestCase):
    def test_equal_models_accept_every_draft(self) -> None:
        model = FixedModel([2.0, 1.0, 0.0])
        result = speculative_decode(
            (),
            model,
            model,
            SpeculativeDecodingConfig(
                draft_tokens=3,
                max_new_tokens=8,
                seed=1,
            ),
        )
        self.assertEqual(len(result.token_ids), 8)
        self.assertEqual(result.accepted_draft_tokens, result.drafted_tokens)
        self.assertLess(result.target_calls, len(result.token_ids))

    def test_rejection_samples_from_target_residual(self) -> None:
        draft = FixedModel([100.0, -100.0])
        target = FixedModel([-100.0, 100.0])
        result = speculative_decode(
            (),
            draft,
            target,
            SpeculativeDecodingConfig(
                draft_tokens=1,
                max_new_tokens=1,
                seed=2,
            ),
        )
        self.assertEqual(result.token_ids, (1,))
        self.assertEqual(result.rejected_draft_tokens, 1)


if __name__ == "__main__":
    unittest.main()
