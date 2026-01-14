from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_audio_file(path: Path) -> Tuple[np.ndarray, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(path, always_2d=True, dtype="float32")
        return audio, int(sr)
    except Exception:
        pass

    try:
        from pydub import AudioSegment  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to decode audio. For MP3, install ffmpeg and `pydub`."
        ) from exc

    seg = AudioSegment.from_file(path)
    sr = int(seg.frame_rate)

    samples = np.array(seg.get_array_of_samples())
    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)

    max_int = float(1 << (8 * seg.sample_width - 1))
    audio = (samples.astype(np.float32) / max_int).reshape((-1, 1))
    return audio, sr

