"""Numerically stable helpers without a NumPy dependency."""

from __future__ import annotations

import math
from collections.abc import Sequence


def softmax(values: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    if not values:
        raise ValueError("softmax requires at least one value")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [float(value) / temperature for value in values]
    peak = max(scaled)
    exponentials = [math.exp(value - peak) for value in scaled]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def argmax(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("argmax requires at least one value")
    return max(range(len(values)), key=values.__getitem__)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
