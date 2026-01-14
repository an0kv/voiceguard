from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceResult:
    p_fake: float
    p_fake_smooth: float
    confidence: float
    is_speech: bool
    indicators: dict[str, float]
    reasons: list[str]


@dataclass(frozen=True)
class WindowScore:
    t_start: float
    t_end: float
    result: InferenceResult

