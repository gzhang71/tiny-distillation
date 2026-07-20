"""Local Hugging Face Transformers teacher implementations."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from tiny_distillation.core.types import TrainingExample
from tiny_distillation.teachers.base import (
    LabelProjector,
    Teacher,
    TeacherResponse,
)


class TransformersTeacher(Teacher):
    """Shared lazy-loading base for local Transformers models."""

    provider = "huggingface"

    def __init__(
        self,
        model: str,
        labels: Sequence[str | int],
        *,
        tokenizer: Any | None = None,
        model_instance: Any | None = None,
        label_projector: LabelProjector | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        device: str | None = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if (tokenizer is None) != (model_instance is None):
            raise ValueError("tokenizer and model_instance must be supplied together")
        self.tokenizer = tokenizer
        self.model_instance = model_instance
        self.device = device
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.generation_kwargs = dict(generation_kwargs or {})
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
        self._ensure_loaded()
        text = self._generate_text(
            self.build_user_prompt(
                example,
                include_reasoning=include_reasoning,
            )
        )
        return self.parse_response(
            text,
            include_reasoning=include_reasoning,
            metadata={"local": True},
        )

    def _ensure_loaded(self) -> None:
        if self.tokenizer is not None and self.model_instance is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Local teachers require `pip install tiny-distillation[hf]`"
            ) from exc
        model_class = self._model_class()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            **self.tokenizer_kwargs,
        )
        self.model_instance = model_class.from_pretrained(
            self.model,
            **self.model_kwargs,
        )
        if self.device is not None:
            self.model_instance.to(self.device)

    @abstractmethod
    def _model_class(self) -> Any:
        ...

    @abstractmethod
    def _generate_text(self, prompt: str) -> str:
        ...

    def _generation_options(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_tokens,
            "do_sample": False,
            **self.generation_kwargs,
        }

    def _move_to_model_device(self, inputs: Any) -> Any:
        device = self.device or getattr(self.model_instance, "device", None)
        return inputs.to(device) if device is not None and hasattr(inputs, "to") else inputs


class HuggingFaceCausalTeacher(TransformersTeacher):
    """Teacher for decoder-only chat models such as Llama and Qwen."""

    def _model_class(self) -> Any:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM

    def _generate_text(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_to_model_device(inputs)
        with torch.no_grad():
            output = self.model_instance.generate(
                **inputs,
                **self._generation_options(),
            )
        sequences = getattr(output, "sequences", output)
        prompt_length = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(
            sequences[0][prompt_length:],
            skip_special_tokens=True,
        )


class LlamaTeacher(HuggingFaceCausalTeacher):
    """Local Meta Llama instruct-model example."""

    def __init__(
        self,
        labels: Sequence[str | int],
        *,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, labels, **kwargs)


class Qwen35Teacher(HuggingFaceCausalTeacher):
    """Local Qwen3.5 text-generation example."""

    def __init__(
        self,
        labels: Sequence[str | int],
        *,
        model: str = "Qwen/Qwen3.5-2B",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, labels, **kwargs)


class T5Teacher(TransformersTeacher):
    """Local encoder-decoder T5/FLAN-T5 example."""

    def __init__(
        self,
        labels: Sequence[str | int],
        *,
        model: str = "google/flan-t5-base",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, labels, **kwargs)

    def _model_class(self) -> Any:
        from transformers import AutoModelForSeq2SeqLM

        return AutoModelForSeq2SeqLM

    def _generate_text(self, prompt: str) -> str:
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
        )
        inputs = self._move_to_model_device(inputs)
        with torch.no_grad():
            output = self.model_instance.generate(
                **inputs,
                **self._generation_options(),
            )
        sequences = getattr(output, "sequences", output)
        return self.tokenizer.decode(sequences[0], skip_special_tokens=True)
