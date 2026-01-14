from __future__ import annotations

import numpy as np

from voiceguard.vad import rms_db


def zero_crossing_rate(audio: np.ndarray) -> float:
    audio = audio.astype(np.float32, copy=False)
    if audio.size < 2:
        return 0.0
    signs = np.signbit(audio)
    return float(np.mean(signs[1:] != signs[:-1]))


def spectral_indicators(
    audio: np.ndarray, sample_rate: int, *, hf_cut_hz: float = 6000.0, rolloff: float = 0.85
) -> dict[str, float]:
    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "hf_energy_ratio": 0.0,
            "spectral_flatness": 0.0,
        }

    n = int(audio.size)
    window = np.hanning(n).astype(np.float32, copy=False)
    xw = audio * window

    power = np.square(np.abs(np.fft.rfft(xw))).astype(np.float32, copy=False)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate)).astype(np.float32, copy=False)

    total = float(np.sum(power)) + 1e-12
    centroid = float(np.sum(freqs * power) / total)
    bandwidth = float(np.sqrt(np.sum(np.square(freqs - centroid) * power) / total))

    cumsum = np.cumsum(power, dtype=np.float64)
    target = float(rolloff) * float(cumsum[-1] if cumsum.size else 0.0)
    rolloff_idx = int(np.searchsorted(cumsum, target, side="left")) if cumsum.size else 0
    rolloff_idx = int(np.clip(rolloff_idx, 0, freqs.size - 1))
    rolloff_hz = float(freqs[rolloff_idx]) if freqs.size else 0.0

    hf_ratio = float(np.sum(power[freqs >= float(hf_cut_hz)]) / total)

    eps = 1e-12
    flatness = float(np.exp(np.mean(np.log(power + eps))) / (np.mean(power + eps)))
    flatness = float(np.clip(flatness, 0.0, 1.0))

    return {
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff_hz": rolloff_hz,
        "hf_energy_ratio": hf_ratio,
        "spectral_flatness": flatness,
    }


def extract_indicators(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    indicators: dict[str, float] = {}
    indicators["rms_db"] = float(rms_db(audio))
    indicators["zcr"] = float(zero_crossing_rate(audio))
    indicators.update(spectral_indicators(audio, sample_rate))
    return indicators

