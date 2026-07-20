"""Anthropic Messages API teacher."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tiny_distillation.core.types import TrainingExample
from tiny_distillation.teachers.base import (
    LabelProjector,
    Teacher,
    TeacherResponse,
)


class ClaudeTeacher(Teacher):
    """Teacher backed by Claude through the Anthropic Messages API."""

    provider = "anthropic"

    def __init__(
        self,
        labels: Sequence[str | int],
        *,
        model: str = "claude-opus-4-8",
        client: Any | None = None,
        api_key: str | None = None,
        label_projector: LabelProjector | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ImportError(
                    "ClaudeTeacher requires `pip install tiny-distillation[api]`"
                ) from exc
            client = Anthropic(api_key=api_key) if api_key is not None else Anthropic()
        self.client = client
        super().__init__(
            model,
            labels,
            label_projector=label_projector,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

    def _request(
        self,
        example: TrainingExample,
        *,
        include_reasoning: bool,
        candidate_index: int,
    ) -> TeacherResponse:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": self.build_user_prompt(
                        example,
                        include_reasoning=include_reasoning,
                    ),
                }
            ],
        )
        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        return self.parse_response(
            text,
            include_reasoning=include_reasoning,
            metadata={"response_id": getattr(message, "id", None)},
        )
