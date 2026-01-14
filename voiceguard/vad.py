from __future__ import annotations

import numpy as np


def rms_db(audio: np.ndarray, eps: float = 1e-12) -> float:
    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return 20.0 * float(np.log10(rms + eps))


def is_speech_window(audio: np.ndarray, threshold_db: float) -> bool:
    return rms_db(audio) > float(threshold_db)

