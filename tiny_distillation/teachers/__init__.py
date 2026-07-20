"""Teacher interfaces and adapters."""

from tiny_distillation.teachers.teacher import (
    CallableTeacher,
    RuleBasedArithmeticTeacher,
    Teacher,
)

__all__ = ["CallableTeacher", "RuleBasedArithmeticTeacher", "Teacher"]
