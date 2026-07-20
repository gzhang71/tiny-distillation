"""Modular distillation objectives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    hard: float = 1.0
    soft: float = 1.0
    cot: float = 0.5


@dataclass(frozen=True)
class DistillationLossOutput:
    total: Tensor
    hard: Tensor
    soft: Tensor
    cot: Tensor


class DistillationLoss(ABC):
    """Computes one selectable combination of distillation objectives."""

    name = "base"

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    @abstractmethod
    def compute(
        self,
        class_logits: Tensor,
        reasoning_logits: Tensor,
        batch: dict[str, Tensor],
        *,
        soft_temperature: float,
        weights: LossWeights,
    ) -> DistillationLossOutput:
        """Compute total and component losses."""

    def _components(
        self,
        class_logits: Tensor,
        reasoning_logits: Tensor,
        batch: dict[str, Tensor],
        soft_temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sample_weights = batch["weights"]
        hard_per_item = F.cross_entropy(
            class_logits,
            batch["hard_labels"],
            reduction="none",
        )
        hard_loss = _weighted_mean(hard_per_item, sample_weights)

        teacher_probabilities = batch["soft_labels"].clamp_min(1e-12)
        teacher_probabilities = teacher_probabilities / teacher_probabilities.sum(
            dim=-1,
            keepdim=True,
        )
        if soft_temperature != 1.0:
            teacher_probabilities = F.softmax(
                teacher_probabilities.log() / soft_temperature,
                dim=-1,
            )
        soft_per_item = F.kl_div(
            F.log_softmax(class_logits / soft_temperature, dim=-1),
            teacher_probabilities,
            reduction="none",
        ).sum(dim=-1) * soft_temperature**2
        soft_loss = _weighted_mean(soft_per_item, sample_weights)

        token_losses = F.cross_entropy(
            reasoning_logits.transpose(1, 2),
            batch["reasoning_target_ids"],
            reduction="none",
            ignore_index=self.pad_id,
        )
        token_mask = batch["reasoning_token_mask"]
        cot_per_item = (token_losses * token_mask).sum(dim=1) / token_mask.sum(
            dim=1
        ).clamp_min(1)
        cot_loss = _weighted_mean(cot_per_item, sample_weights)
        return hard_loss, soft_loss, cot_loss

    @staticmethod
    def _output(
        total: Tensor,
        components: tuple[Tensor, Tensor, Tensor],
    ) -> DistillationLossOutput:
        return DistillationLossOutput(total, *components)


class HardCrossEntropyLoss(DistillationLoss):
    name = "hard_ce"

    def compute(
        self,
        class_logits: Tensor,
        reasoning_logits: Tensor,
        batch: dict[str, Tensor],
        *,
        soft_temperature: float,
        weights: LossWeights,
    ) -> DistillationLossOutput:
        components = self._components(
            class_logits,
            reasoning_logits,
            batch,
            soft_temperature,
        )
        return self._output(weights.hard * components[0], components)


class SoftKLDistillationLoss(DistillationLoss):
    name = "soft_kl"

    def compute(
        self,
        class_logits: Tensor,
        reasoning_logits: Tensor,
        batch: dict[str, Tensor],
        *,
        soft_temperature: float,
        weights: LossWeights,
    ) -> DistillationLossOutput:
        components = self._components(
            class_logits,
            reasoning_logits,
            batch,
            soft_temperature,
        )
        return self._output(weights.soft * components[1], components)


class CoTTokenCrossEntropyLoss(DistillationLoss):
    name = "cot_token_ce"

    def compute(
        self,
        class_logits: Tensor,
        reasoning_logits: Tensor,
        batch: dict[str, Tensor],
        *,
        soft_temperature: float,
        weights: LossWeights,
    ) -> DistillationLossOutput:
        components = self._components(
            class_logits,
            reasoning_logits,
            batch,
            soft_temperature,
        )
        return self._output(weights.cot * components[2], components)


class CombinedDistillationLoss(DistillationLoss):
    name = "combined"

    def compute(
        self,
        class_logits: Tensor,
        reasoning_logits: Tensor,
        batch: dict[str, Tensor],
        *,
        soft_temperature: float,
        weights: LossWeights,
    ) -> DistillationLossOutput:
        components = self._components(
            class_logits,
            reasoning_logits,
            batch,
            soft_temperature,
        )
        total = (
            weights.hard * components[0]
            + weights.soft * components[1]
            + weights.cot * components[2]
        )
        return self._output(total, components)


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)
