from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if n_mels <= 0:
        raise ValueError("n_mels must be > 0")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")
    if not (0.0 <= fmin < fmax <= sample_rate / 2):
        raise ValueError("Expected 0 <= fmin < fmax <= sample_rate/2")

    fft_freqs = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1, dtype=np.float64)

    mels = np.linspace(
        _hz_to_mel(np.array([fmin], dtype=np.float64))[0],
        _hz_to_mel(np.array([fmax], dtype=np.float64))[0],
        n_mels + 2,
        dtype=np.float64,
    )
    hz = _mel_to_hz(mels)

    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if right <= left:
            continue
        if center > left:
            fb[i, left:center] = (fft_freqs[left:center] - fft_freqs[left]) / (
                fft_freqs[center] - fft_freqs[left] + 1e-12
            )
        if right > center:
            fb[i, center:right] = (fft_freqs[right] - fft_freqs[center:right]) / (
                fft_freqs[right] - fft_freqs[center] + 1e-12
            )

    return fb.astype(dtype)


@dataclass(frozen=True)
class LogMelSpecParams:
    sample_rate: int
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    fmin: float
    fmax: float


def log_mel_spectrogram(audio: np.ndarray, params: LogMelSpecParams) -> np.ndarray:
    return log_mel_spectrogram_with_filterbank(audio, params, filterbank=None)


def log_mel_spectrogram_with_filterbank(
    audio: np.ndarray, params: LogMelSpecParams, *, filterbank: np.ndarray | None
) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros((params.n_mels, 0), dtype=np.float32)

    try:
        from scipy.signal import get_window, stft  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for STFT") from exc

    window = get_window("hann", params.win_length, fftbins=True)
    n_overlap = max(0, params.win_length - params.hop_length)

    _, _, zxx = stft(
        audio,
        fs=params.sample_rate,
        nperseg=params.win_length,
        noverlap=n_overlap,
        nfft=params.n_fft,
        window=window,
        boundary=None,
        padded=False,
    )
    power = np.square(np.abs(zxx)).astype(np.float32, copy=False)  # (freq, frames)

    fb = filterbank
    if fb is None:
        fb = mel_filterbank(
            sample_rate=params.sample_rate,
            n_fft=params.n_fft,
            n_mels=params.n_mels,
            fmin=params.fmin,
            fmax=params.fmax,
        )
    mel = np.matmul(fb, power)  # (n_mels, frames)
    return np.log(mel + 1e-10).astype(np.float32, copy=False)
