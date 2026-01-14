from __future__ import annotations

import numpy as np

from voiceguard.dsp.resample import resample_audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim != 2:
        raise ValueError(f"Expected 1D/2D audio array, got shape={audio.shape}")
    return audio.mean(axis=1).astype(np.float32, copy=False)


def normalize_audio(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    audio = audio - float(np.mean(audio)) if audio.size else audio
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs <= 0:
        return audio
    return (audio / max_abs) * float(peak)


def preprocess_audio(
    audio: np.ndarray, orig_sr: int, target_sr: int, *, normalize: bool = True
) -> np.ndarray:
    audio_mono = to_mono(audio)
    audio_rs = resample_audio(audio_mono, orig_sr=orig_sr, target_sr=target_sr)
    return normalize_audio(audio_rs) if normalize else audio_rs
