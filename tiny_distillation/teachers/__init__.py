"""Teacher interfaces and adapters."""

from tiny_distillation.teachers.anthropic_teacher import ClaudeTeacher
from tiny_distillation.teachers.base import (
    LabelProjector,
    Teacher,
    TeacherResponse,
)
from tiny_distillation.teachers.callable import CallableTeacher
from tiny_distillation.teachers.deepseek_teacher import DeepSeekTeacher
from tiny_distillation.teachers.huggingface_teacher import (
    HuggingFaceCausalTeacher,
    LlamaTeacher,
    Qwen35Teacher,
    T5Teacher,
    TransformersTeacher,
)
from tiny_distillation.teachers.openai_teacher import ChatGPTTeacher
from tiny_distillation.teachers.rule_based import RuleBasedArithmeticTeacher

__all__ = [
    "CallableTeacher",
    "ChatGPTTeacher",
    "ClaudeTeacher",
    "DeepSeekTeacher",
    "HuggingFaceCausalTeacher",
    "LabelProjector",
    "LlamaTeacher",
    "Qwen35Teacher",
    "RuleBasedArithmeticTeacher",
    "T5Teacher",
    "Teacher",
    "TeacherResponse",
    "TransformersTeacher",
]
