from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

from voiceguard.config import EnhanceConfig


class NoiseReducer:
    def __init__(self, *, strength: float, ema: float) -> None:
        self._strength = float(strength)
        self._ema = float(ema)
        self._noise_mag: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self._window_len: int = 0

    def reset(self) -> None:
        self._noise_mag = None
        self._window = None
        self._window_len = 0

    def update_profile(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        mag = self._fft_mag(audio)
        if self._noise_mag is None or self._noise_mag.shape != mag.shape:
            self._noise_mag = mag
        else:
            self._noise_mag = (1.0 - self._ema) * self._noise_mag + self._ema * mag

    def reduce(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0 or self._noise_mag is None:
            return audio
        spec = self._fft(audio)
        mag = np.abs(spec)
        phase = np.exp(1j * np.angle(spec))
        noise_mag = self._noise_mag
        if noise_mag is None or noise_mag.shape != mag.shape:
            return audio
        cleaned = np.maximum(0.0, mag - noise_mag * self._strength)
        spec_clean = cleaned * phase
        out = np.fft.irfft(spec_clean, n=audio.size)
        return out.astype(np.float32, copy=False)

    def _fft(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float32, copy=False)
        window = self._get_window(x.size)
        return np.fft.rfft(x * window)

    def _fft_mag(self, audio: np.ndarray) -> np.ndarray:
        spec = self._fft(audio)
        return np.abs(spec)

    def _get_window(self, n: int) -> np.ndarray:
        if self._window is None or self._window_len != n:
            self._window = np.hanning(n).astype(np.float32, copy=False)
            self._window_len = int(n)
        return self._window


@dataclass
class AudioEnhancer:
    config: EnhanceConfig
    sample_rate: int

    def __post_init__(self) -> None:
        self._noise = NoiseReducer(
            strength=float(self.config.noise_strength),
            ema=float(self.config.noise_ema),
        )
        self._sos: Optional[np.ndarray] = None
        self._sos_state: Optional[np.ndarray] = None
        if self.config.bandpass:
            self._sos = self._design_bandpass()

    def reset(self) -> None:
        self._noise.reset()
        self._sos_state = None

    def update_noise(self, audio: np.ndarray) -> None:
        if not self.config.noise_reduction:
            return
        x = self._apply_bandpass(audio) if self.config.bandpass else audio
        self._noise.update_profile(x)

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float32, copy=False)
        if self.config.bandpass:
            x = self._apply_bandpass(x)
        if self.config.noise_reduction:
            x = self._noise.reduce(x)
        return x.astype(np.float32, copy=False)

    def _design_bandpass(self) -> np.ndarray:
        nyq = float(self.sample_rate) * 0.5
        low = max(float(self.config.bandpass_low_hz), 20.0) / nyq
        high = min(float(self.config.bandpass_high_hz), nyq * 0.98) / nyq
        low = min(max(low, 0.001), 0.99)
        high = min(max(high, low + 0.01), 0.999)
        return signal.butter(4, [low, high], btype="bandpass", output="sos")

    def _apply_bandpass(self, audio: np.ndarray) -> np.ndarray:
        if self._sos is None:
            return audio.astype(np.float32, copy=False)
        x = audio.astype(np.float32, copy=False)
        if self._sos_state is None:
            zi = signal.sosfilt_zi(self._sos)
            self._sos_state = zi * (float(x[0]) if x.size else 0.0)
        x, self._sos_state = signal.sosfilt(self._sos, x, zi=self._sos_state)
        return x.astype(np.float32, copy=False)
