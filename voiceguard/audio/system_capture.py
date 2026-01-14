from __future__ import annotations

import queue
from typing import Optional

import numpy as np

from voiceguard.audio.chunk import AudioChunk


class SystemAudioCapture:
    def __init__(self, device: Optional[int] = None, *, loopback: bool = True) -> None:
        self._device = device
        self._loopback_requested = bool(loopback)
        self._loopback_active = False
        self._stream = None
        self._queue: "queue.Queue[AudioChunk]" = queue.Queue(maxsize=64)
        self._running = False

    @property
    def queue(self) -> "queue.Queue[AudioChunk]":
        return self._queue

    @property
    def loopback_active(self) -> bool:
        return bool(self._loopback_active)

    def _is_wasapi_device(self, sd, device: Optional[int]) -> bool:
        try:
            dev_info = sd.query_devices(device)
            hostapi_idx = int(dev_info.get("hostapi", -1))
            hostapi = sd.query_hostapis()[hostapi_idx] if hostapi_idx >= 0 else {}
            name = str(hostapi.get("name", "")).upper()
            return "WASAPI" in name
        except Exception:
            return False

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
                pass
            if indata.ndim == 1:
                mono = indata.astype(np.float32, copy=True)
            else:
                if indata.shape[1] == 1:
                    mono = indata[:, 0].astype(np.float32, copy=True)
                else:
                    mono = np.mean(indata, axis=1).astype(np.float32, copy=True)
            try:
                self._queue.put_nowait(AudioChunk(samples=mono, sample_rate=int(stream_sr)))
            except queue.Full:
                pass

        stream_sr = int(preferred_sample_rate)

        device = self._device
        if device is None and self._loopback_requested:
            try:
                default_out = sd.default.device[1]
                device = int(default_out) if default_out is not None and int(default_out) >= 0 else None
            except Exception:
                device = None

        use_loopback = bool(
            self._loopback_requested
            and hasattr(sd, "WasapiSettings")
            and self._is_wasapi_device(sd, device)
        )
        self._loopback_active = bool(use_loopback)

        if not use_loopback and self._loopback_requested and self._device is None:
            device = None

        kind = "output" if use_loopback else "input"
        try:
            dev_info = sd.query_devices(device, kind)
        except Exception:
            dev_info = {}

        default_sr = int(dev_info.get("default_samplerate", stream_sr))
        if use_loopback:
            max_channels = int(dev_info.get("max_output_channels", 2))
        else:
            max_channels = int(dev_info.get("max_input_channels", 1))
        channels = max(1, min(2, max_channels))

        extra_settings = sd.WasapiSettings(loopback=True) if use_loopback else None

        for sr in (stream_sr, default_sr):
            try:
                blocksize = max(0, int(sr * float(block_sec)))
                self._stream = sd.InputStream(
                    samplerate=sr,
                    device=device,
                    channels=channels,
                    dtype="float32",
                    blocksize=blocksize,
                    callback=callback,
                    extra_settings=extra_settings,
                )
                self._stream.start()
                stream_sr = int(sr)
                break
            except Exception:
                self._stream = None
                continue

        if self._stream is None:  # pragma: no cover
            self._running = False
            raise RuntimeError("Failed to start system audio capture (sounddevice loopback).")

        return int(stream_sr)

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
