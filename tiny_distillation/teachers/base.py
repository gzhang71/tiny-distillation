"""Abstract teacher contract shared by hosted and local model backends."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tiny_distillation.core.math_utils import clamp, softmax
from tiny_distillation.core.types import TeacherPrediction, TrainingExample

LabelProjector = Callable[
    ["TeacherResponse", TrainingExample, tuple[str, ...]],
    Sequence[float],
]


@dataclass(frozen=True)
class TeacherResponse:
    """Provider-independent structured generation."""

    answer: str
    reasoning: str = ""
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Teacher(ABC):
    """Base class that converts backend text into distillation targets."""

    provider = "unknown"
    default_system_prompt = (
        "You are a teacher creating high-quality supervised training data. "
        "Return only a JSON object with string fields 'answer' and 'reasoning' "
        "and a numeric 'confidence' from 0 to 1. The answer must be exactly one "
        "of the supplied labels. Give a concise, self-contained justification; "
        "do not reveal private or hidden internal reasoning."
    )

    def __init__(
        self,
        model: str,
        labels: Sequence[str | int],
        *,
        label_projector: LabelProjector | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        if not labels:
            raise ValueError("teacher labels cannot be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.model = model
        self.labels = tuple(str(label) for label in labels)
        self.label_projector = label_projector
        self.system_prompt = system_prompt or self.default_system_prompt
        self.max_tokens = max_tokens

    def generate(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool = True,
        candidate_index: int = 0,
    ) -> TeacherPrediction:
        response = self._request(
            example,
            include_reasoning=include_reasoning,
            candidate_index=candidate_index,
        )
        logits = self.project_logits(response, example)
        probabilities = softmax(logits)
        metadata = {
            "provider": self.provider,
            "model": self.model,
            **dict(response.metadata),
        }
        return TeacherPrediction(
            example_id=example.id,
            prompt=example.prompt,
            answer=response.answer,
            reasoning=response.reasoning if include_reasoning else "",
            logits=logits,
            confidence=max(probabilities),
            candidate_index=candidate_index,
            metadata=metadata,
        )

    @abstractmethod
    def _request(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool,
        candidate_index: int,
    ) -> TeacherResponse:
        """Call a backend and return a provider-independent response."""

    def build_user_prompt(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool,
    ) -> str:
        reasoning_instruction = (
            "Include a concise justification in 'reasoning'."
            if include_reasoning
            else "Set 'reasoning' to an empty string."
        )
        return (
            f"Labels: {json.dumps(self.labels)}\n"
            f"{reasoning_instruction}\n"
            f"Question: {example.prompt}"
        )

    def parse_response(
        self,
        text: str,
        *,
        include_reasoning: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> TeacherResponse:
        payload = self._extract_json(text)
        answer = str(payload.get("answer", "")).strip()
        if not answer:
            raise ValueError("teacher response is missing a non-empty 'answer'")
        reasoning = str(payload.get("reasoning", "")).strip()
        confidence = clamp(float(payload.get("confidence", 1.0)))
        return TeacherResponse(
            answer=answer,
            reasoning=reasoning if include_reasoning else "",
            confidence=confidence,
            metadata=metadata or {},
        )

    def project_logits(
        self,
        response: TeacherResponse,
        example: TrainingExample,
    ) -> tuple[float, ...]:
        if self.label_projector is not None:
            logits = tuple(
                float(value)
                for value in self.label_projector(response, example, self.labels)
            )
            if len(logits) != len(self.labels):
                raise ValueError(
                    "label_projector must return one logit per teacher label"
                )
            if not all(math.isfinite(value) for value in logits):
                raise ValueError("label_projector returned a non-finite logit")
            return logits

        normalized_answer = self._normalize(response.answer)
        matching = [
            index
            for index, label in enumerate(self.labels)
            if self._normalize(label) == normalized_answer
        ]
        if not matching:
            raise ValueError(
                f"teacher answer {response.answer!r} does not match any configured "
                "label; return an exact label or provide label_projector"
            )
        if len(self.labels) == 1:
            return (0.0,)

        minimum_confidence = 1.0 / len(self.labels) + 1e-6
        confidence = clamp(
            response.confidence,
            lower=minimum_confidence,
            upper=1.0 - 1e-6,
        )
        other_probability = (1.0 - confidence) / (len(self.labels) - 1)
        probabilities = [other_probability] * len(self.labels)
        probabilities[matching[0]] = confidence
        return tuple(math.log(max(probability, 1e-12)) for probability in probabilities)

    @staticmethod
    def _extract_json(text: str) -> Mapping[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            cleaned = cleaned[first_newline + 1 :] if first_newline >= 0 else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("teacher response does not contain a JSON object")
        try:
            payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("teacher response contains invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("teacher response JSON must be an object")
        return payload

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.strip().lower().split())
