"""DeepSeek OpenAI-compatible API teacher."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from tiny_distillation.core.types import TrainingExample
from tiny_distillation.teachers.base import (
    LabelProjector,
    Teacher,
    TeacherResponse,
)


class DeepSeekTeacher(Teacher):
    """Teacher backed by DeepSeek's Chat Completions endpoint."""

    provider = "deepseek"

    def __init__(
        self,
        labels: Sequence[str | int],
        *,
        model: str = "deepseek-v4-pro",
        client: Any | None = None,
        api_key: str | None = None,
        label_projector: LabelProjector | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        thinking: bool = False,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "DeepSeekTeacher requires `pip install tiny-distillation[api]`"
                ) from exc
            client = OpenAI(
                api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
        self.client = client
        self.thinking = thinking
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
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.build_user_prompt(
                        example,
                        include_reasoning=include_reasoning,
                    ),
                },
            ],
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.thinking:
            request["reasoning_effort"] = "high"
            request["extra_body"] = {"thinking": {"type": "enabled"}}
        completion = self.client.chat.completions.create(**request)
        message = completion.choices[0].message
        response = self.parse_response(
            message.content,
            include_reasoning=include_reasoning,
            metadata={"response_id": getattr(completion, "id", None)},
        )
        reasoning_content = getattr(message, "reasoning_content", None)
        if include_reasoning and reasoning_content and not response.reasoning:
            return TeacherResponse(
                answer=response.answer,
                reasoning=str(reasoning_content),
                confidence=response.confidence,
                metadata=response.metadata,
            )
        return response
