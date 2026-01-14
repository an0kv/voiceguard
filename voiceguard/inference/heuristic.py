from __future__ import annotations

import math
from typing import Mapping, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def heuristic_p_fake(indicators: Mapping[str, float]) -> Tuple[float, list[str]]:
    hf_ratio = float(indicators.get("hf_energy_ratio", 0.0))
    rolloff_hz = float(indicators.get("spectral_rolloff_hz", 0.0))
    flatness = float(indicators.get("spectral_flatness", 0.0))

    # Very rough, demo-only heuristic:
    # - many synthetic/over-compressed samples show reduced HF energy / rolloff
    # - overly "sterile" spectrum can have low flatness
    cutoff_score = _clamp((0.06 - hf_ratio) / 0.06, 0.0, 1.0)
    rolloff_score = _clamp((4200.0 - rolloff_hz) / 4200.0, 0.0, 1.0)
    flat_score = _clamp((0.12 - flatness) / 0.12, 0.0, 1.0)

    raw = 0.50 * cutoff_score + 0.35 * rolloff_score + 0.15 * flat_score
    p_fake = _sigmoid((raw - 0.35) * 7.0)

    reasons = heuristic_reasons(indicators)

    return float(_clamp(p_fake, 0.0, 1.0)), reasons


def heuristic_reasons(indicators: Mapping[str, float]) -> list[str]:
    hf_ratio = float(indicators.get("hf_energy_ratio", 0.0))
    rolloff_hz = float(indicators.get("spectral_rolloff_hz", 0.0))
    flatness = float(indicators.get("spectral_flatness", 0.0))

    reasons: list[str] = []
    if hf_ratio < 0.02:
        reasons.append("низкая доля энергии в ВЧ (возможен срез/кодек/TTS)")
    if rolloff_hz < 3200.0:
        reasons.append("низкий spectral roll-off (возможен срез ВЧ)")
    if flatness < 0.08:
        reasons.append("низкая spectral flatness (слишком 'стерильно/тонально')")
    return reasons
