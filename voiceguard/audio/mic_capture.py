from __future__ import annotations

import queue
from typing import Optional

import numpy as np

from voiceguard.audio.chunk import AudioChunk


class MicCapture:
    def __init__(self, device: Optional[int] = None) -> None:
        self._device = device
        self._stream = None
        self._queue: "queue.Queue[AudioChunk]" = queue.Queue(maxsize=64)
        self._running = False

    @property
    def queue(self) -> "queue.Queue[AudioChunk]":
        return self._queue

    def start(self, preferred_sample_rate: int, block_sec: float = 0.10) -> int:
        import sounddevice as sd  # type: ignore

        if self._running:
            return int(preferred_sample_rate)

        self._queue.queue.clear()  # type: ignore[attr-defined]
        self._running = True

        def callback(indata: np.ndarray, frames: int, time, status) -> None:  # noqa: ANN001
            if not self._running:
                return
            if status:
                # Drop status-only info; keep audio.
                pass
            mono = indata[:, 0].astype(np.float32, copy=True)
            try:
                self._queue.put_nowait(AudioChunk(samples=mono, sample_rate=int(stream_sr)))
            except queue.Full:
                # Backpressure: drop chunk to keep latency bounded.
                pass

        # Prefer config SR, but fallback to device default if unsupported.
        stream_sr = int(preferred_sample_rate)
        try:
            dev_info = sd.query_devices(self._device, "input")
            default_sr = int(dev_info.get("default_samplerate", stream_sr))
        except Exception:
            default_sr = stream_sr

        for sr in (stream_sr, default_sr):
            try:
                blocksize = max(0, int(sr * float(block_sec)))
                self._stream = sd.InputStream(
                    samplerate=sr,
                    device=self._device,
                    channels=1,
                    dtype="float32",
                    blocksize=blocksize,
                    callback=callback,
                )
                self._stream.start()
                stream_sr = int(sr)
                break
            except Exception:
                self._stream = None
                continue

        if self._stream is None:  # pragma: no cover
            self._running = False
            raise RuntimeError("Failed to start microphone capture (sounddevice).")

        return int(stream_sr)

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
