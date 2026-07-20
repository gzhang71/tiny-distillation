"""OpenAI Responses API teacher."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tiny_distillation.core.types import TrainingExample
from tiny_distillation.teachers.base import (
    LabelProjector,
    Teacher,
    TeacherResponse,
)


class ChatGPTTeacher(Teacher):
    """Teacher backed by an OpenAI model through the Responses API."""

    provider = "openai"

    def __init__(
        self,
        labels: Sequence[str | int],
        *,
        model: str = "gpt-5.6",
        client: Any | None = None,
        api_key: str | None = None,
        label_projector: LabelProjector | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "ChatGPTTeacher requires `pip install tiny-distillation[api]`"
                ) from exc
            client = OpenAI(api_key=api_key) if api_key is not None else OpenAI()
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
        response = self.client.responses.create(
            model=self.model,
            instructions=self.system_prompt,
            input=self.build_user_prompt(
                example,
                include_reasoning=include_reasoning,
            ),
            max_output_tokens=self.max_tokens,
        )
        return self.parse_response(
            response.output_text,
            include_reasoning=include_reasoning,
            metadata={"response_id": getattr(response, "id", None)},
        )
