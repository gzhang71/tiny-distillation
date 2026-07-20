"""Exact speculative sampling with a small draft model and target verifier."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol

import torch
from torch import Tensor


class AutoregressiveModel(Protocol):
    def next_token_logits(self, token_ids: tuple[int, ...]) -> Tensor:
        ...


@dataclass(frozen=True)
class SpeculativeDecodingConfig:
    draft_tokens: int = 4
    max_new_tokens: int = 32
    temperature: float = 1.0
    eos_token_id: int | None = None
    seed: int = 7

    def __post_init__(self) -> None:
        if self.draft_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("draft_tokens and max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class SpeculativeDecodingResult:
    token_ids: tuple[int, ...]
    accepted_draft_tokens: int
    rejected_draft_tokens: int
    drafted_tokens: int
    target_calls: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / max(1, self.drafted_tokens)


def speculative_decode(
    prefix: tuple[int, ...],
    draft_model: AutoregressiveModel,
    target_model: AutoregressiveModel,
    config: SpeculativeDecodingConfig | None = None,
) -> SpeculativeDecodingResult:
    """Sample from the target distribution using draft proposals.

    The residual distribution on rejection preserves exact target-model sampling.
    """
    config = config or SpeculativeDecodingConfig()
    generator = torch.Generator().manual_seed(config.seed)
    output = list(prefix)
    initial_length = len(output)
    accepted = rejected = drafted = target_calls = 0

    while len(output) - initial_length < config.max_new_tokens:
        remaining = config.max_new_tokens - (len(output) - initial_length)
        proposal_count = min(config.draft_tokens, remaining)
        draft_prefix = list(output)
        proposals: list[tuple[int, Tensor]] = []
        for _ in range(proposal_count):
            draft_probabilities = _probabilities(
                draft_model.next_token_logits(tuple(draft_prefix)),
                config.temperature,
            )
            token = _sample(draft_probabilities, generator)
            proposals.append((token, draft_probabilities))
            draft_prefix.append(token)
            drafted += 1
            if token == config.eos_token_id:
                break

        verification_prefixes = [
            tuple(output + [proposal[0] for proposal in proposals[:index]])
            for index in range(len(proposals) + 1)
        ]
        target_distributions, calls = _target_probabilities(
            target_model,
            verification_prefixes,
            config.temperature,
        )
        target_calls += calls

        rejected_block = False
        for index, (token, draft_probabilities) in enumerate(proposals):
            target_probabilities = target_distributions[index]
            acceptance_probability = min(
                1.0,
                float(target_probabilities[token] / draft_probabilities[token].clamp_min(1e-12)),
            )
            if float(torch.rand((), generator=generator)) <= acceptance_probability:
                output.append(token)
                accepted += 1
            else:
                residual = (target_probabilities - draft_probabilities).clamp_min(0)
                if float(residual.sum()) <= 1e-12:
                    residual = target_probabilities
                replacement = _sample(residual / residual.sum(), generator)
                output.append(replacement)
                rejected += 1
                rejected_block = True
            if output[-1] == config.eos_token_id:
                return _result(output, initial_length, accepted, rejected, drafted, target_calls)
            if rejected_block or len(output) - initial_length >= config.max_new_tokens:
                break

        if rejected_block or len(output) - initial_length >= config.max_new_tokens:
            continue

        output.append(_sample(target_distributions[-1], generator))
        if output[-1] == config.eos_token_id:
            break

    return _result(output, initial_length, accepted, rejected, drafted, target_calls)


def _probabilities(logits: Tensor, temperature: float) -> Tensor:
    values = torch.as_tensor(logits, dtype=torch.float32)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("next_token_logits must return a non-empty 1D tensor")
    return torch.softmax(values / temperature, dim=-1)


def _sample(probabilities: Tensor, generator: torch.Generator) -> int:
    return int(torch.multinomial(probabilities, 1, generator=generator))


def _target_probabilities(
    model: AutoregressiveModel,
    prefixes: Sequence[tuple[int, ...]],
    temperature: float,
) -> tuple[list[Tensor], int]:
    batch_method = getattr(model, "next_token_logits_batch", None)
    if callable(batch_method):
        batch_logits = torch.as_tensor(batch_method(prefixes), dtype=torch.float32)
        if batch_logits.ndim != 2 or batch_logits.shape[0] != len(prefixes):
            raise ValueError(
                "next_token_logits_batch must return [prefixes, vocabulary]"
            )
        return [
            torch.softmax(row / temperature, dim=-1) for row in batch_logits
        ], 1
    return [
        _probabilities(model.next_token_logits(prefix), temperature)
        for prefix in prefixes
    ], len(prefixes)


def _result(
    output: list[int],
    initial_length: int,
    accepted: int,
    rejected: int,
    drafted: int,
    target_calls: int,
) -> SpeculativeDecodingResult:
    return SpeculativeDecodingResult(
        token_ids=tuple(output[initial_length:]),
        accepted_draft_tokens=accepted,
        rejected_draft_tokens=rejected,
        drafted_tokens=drafted,
        target_calls=target_calls,
    )
