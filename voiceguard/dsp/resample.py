from __future__ import annotations

import math

import numpy as np


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    if int(orig_sr) == int(target_sr):
        return audio

    try:
        from scipy.signal import resample_poly  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for resampling") from exc

    orig_sr_i = int(orig_sr)
    target_sr_i = int(target_sr)
    g = math.gcd(orig_sr_i, target_sr_i)
    up = target_sr_i // g
    down = orig_sr_i // g

    if audio.size == 0:
        return audio

    return resample_poly(audio, up=up, down=down).astype(np.float32, copy=False)

