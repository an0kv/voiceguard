from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AudioChunk:
    samples: np.ndarray  # float32 mono
    sample_rate: int
