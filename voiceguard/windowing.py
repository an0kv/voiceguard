from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing import Iterator, Tuple


def _sec_to_samples(sec: float, sample_rate: int) -> int:
    return int(round(float(sec) * float(sample_rate)))


def estimate_num_windows(num_samples: int, *, window_samples: int, hop_samples: int) -> int:
    if num_samples <= 0 or window_samples <= 0 or hop_samples <= 0:
        return 0
    if num_samples < window_samples:
        return 1
    return 1 + int((num_samples - window_samples) // hop_samples)


def window_params(*, sample_rate: int, window_sec: float, hop_sec: float) -> tuple[int, int]:
    window_samples = _sec_to_samples(window_sec, sample_rate)
    hop_samples = _sec_to_samples(hop_sec, sample_rate)
    if window_samples <= 0:
        raise ValueError("window_sec is too small (<= 0 samples)")
    if hop_samples <= 0:
        raise ValueError("hop_sec is too small (<= 0 samples)")
    return window_samples, hop_samples


def iter_windows(
    audio: np.ndarray, *, window_samples: int, hop_samples: int
) -> Iterator[Tuple[int, np.ndarray]]:
    audio = audio.astype(np.float32, copy=False).reshape(-1)
    if audio.size == 0:
        return

    if audio.size < window_samples:
        padded = np.zeros((window_samples,), dtype=np.float32)
        padded[: audio.size] = audio
        yield 0, padded
        return

    for start in range(0, int(audio.size - window_samples + 1), int(hop_samples)):
        yield int(start), audio[int(start) : int(start + window_samples)]


@dataclass(frozen=True)
class StreamWindow:
    start_sample: int
    samples: np.ndarray


class StreamWindowProcessor:
    def __init__(self, *, sample_rate: int, window_sec: float, hop_sec: float) -> None:
        self._sample_rate = int(sample_rate)
        self._window_samples = _sec_to_samples(window_sec, self._sample_rate)
        self._hop_samples = _sec_to_samples(hop_sec, self._sample_rate)
        if self._window_samples <= 0:
            raise ValueError("window_sec is too small (<= 0 samples)")
        if self._hop_samples <= 0:
            raise ValueError("hop_sec is too small (<= 0 samples)")

        self._buffer = np.zeros((0,), dtype=np.float32)
        self._buffer_start = 0  # global sample index of buffer[0]
        self._total_samples = 0
        self._next_window_start = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def window_samples(self) -> int:
        return self._window_samples

    @property
    def hop_samples(self) -> int:
        return self._hop_samples

    def push(self, samples: np.ndarray) -> list[StreamWindow]:
        samples = samples.astype(np.float32, copy=False).reshape(-1)
        if samples.size == 0:
            return []

        self._buffer = np.concatenate([self._buffer, samples])
        self._total_samples += int(samples.size)

        windows: list[StreamWindow] = []
        while self._next_window_start + self._window_samples <= self._total_samples:
            start_offset = self._next_window_start - self._buffer_start
            if start_offset < 0:
                # Shouldn't happen, but keep safe.
                start_offset = 0
                self._next_window_start = self._buffer_start

            end_offset = start_offset + self._window_samples
            if end_offset > self._buffer.size:
                break

            window = self._buffer[start_offset:end_offset]
            windows.append(StreamWindow(start_sample=int(self._next_window_start), samples=window))
            self._next_window_start += int(self._hop_samples)

        # Trim everything before the next window start to keep memory bounded.
        trim = max(0, int(self._next_window_start - self._buffer_start))
        if trim > 0:
            self._buffer = self._buffer[trim:]
            self._buffer_start += trim

        return windows
