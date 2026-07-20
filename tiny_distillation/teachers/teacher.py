"""Compatibility exports for the reorganized teacher package."""

from tiny_distillation.teachers.base import (
    LabelProjector,
    Teacher,
    TeacherResponse,
)
from tiny_distillation.teachers.callable import CallableTeacher
from tiny_distillation.teachers.rule_based import RuleBasedArithmeticTeacher

__all__ = [
    "CallableTeacher",
    "LabelProjector",
    "RuleBasedArithmeticTeacher",
    "Teacher",
    "TeacherResponse",
]
