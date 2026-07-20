"""Shared immutable records passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingExample:
    id: str
    prompt: str
    label: int | None = None
    reference_answer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeacherPrediction:
    example_id: str
    prompt: str
    answer: str
    reasoning: str
    logits: tuple[float, ...]
    confidence: float
    candidate_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredPrediction:
    prediction: TeacherPrediction
    answer_score: float
    reasoning_score: float
    confidence_score: float
    total_score: float
    accepted: bool
    diagnostics: Mapping[str, float | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibratedLabel:
    example_id: str
    prompt: str
    hard_label: int
    soft_labels: tuple[float, ...]
    answer: str
    reasoning: str
    weight: float
    source_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingHistory:
    losses: tuple[float, ...]
    hard_losses: tuple[float, ...]
    soft_losses: tuple[float, ...]
    cot_losses: tuple[float, ...]
    validation_losses: tuple[float, ...] = ()
    learning_rates: tuple[float, ...] = ()
    teacher_forcing_ratios: tuple[float, ...] = ()
    best_epoch: int | None = None
    stopped_early: bool = False
