from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlertSegment:
    start_sec: float
    end_sec: float


class AlertTracker:
    def __init__(self, *, threshold: float, hold_sec: float, step_sec: float) -> None:
        self._threshold = float(threshold)
        self._hold_sec = float(hold_sec)
        self._step_sec = float(step_sec)

        self._above_sec = 0.0
        self._active = False
        self._active_start = 0.0
        self._segments: list[AlertSegment] = []

    @property
    def segments(self) -> list[AlertSegment]:
        return list(self._segments)

    @property
    def active(self) -> bool:
        return bool(self._active)

    def reset(self) -> None:
        self._above_sec = 0.0
        self._active = False
        self._active_start = 0.0
        self._segments.clear()

    def update(self, *, t_start: float, t_end: float, p: float, is_speech: bool) -> bool:
        if not is_speech:
            if self._active:
                self._segments.append(AlertSegment(start_sec=self._active_start, end_sec=float(t_end)))
            self._active = False
            self._above_sec = 0.0
            return False

        if float(p) >= self._threshold:
            self._above_sec += self._step_sec
            if not self._active and self._above_sec >= self._hold_sec:
                self._active = True
                # Approximate the alert start as "current end - accumulated above time".
                self._active_start = max(float(t_start), float(t_end) - self._above_sec)
        else:
            if self._active:
                self._segments.append(AlertSegment(start_sec=self._active_start, end_sec=float(t_end)))
            self._active = False
            self._above_sec = 0.0

        return bool(self._active)

    def finalize(self, *, t_end: float) -> None:
        if self._active:
            self._segments.append(AlertSegment(start_sec=self._active_start, end_sec=float(t_end)))
        self._active = False
        self._above_sec = 0.0

