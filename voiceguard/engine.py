from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from voiceguard.config import AppConfig, EnhanceConfig
from voiceguard.dsp.mel import (
    LogMelSpecParams,
    log_mel_spectrogram_with_filterbank,
    mel_filterbank,
)
from voiceguard.dsp.enhance import AudioEnhancer
from voiceguard.features import extract_indicators
from voiceguard.inference import HfAudioClassifier, OnnxModel, heuristic_p_fake, heuristic_reasons
from voiceguard.preprocess import normalize_audio, preprocess_audio
from voiceguard.types import InferenceResult


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


class VoiceGuardEngine:
    def __init__(self, config: AppConfig, *, base_dir: Optional[Path] = None) -> None:
        self._config = config
        self._base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]

        self._requested_backend = str(config.model.backend).lower()
        self._backend_in_use = "heuristic"
        self._backend_note = ""

        self._target_sr = int(config.sample_rate)
        self._ema_alpha = float(config.smoothing.ema_alpha)
        self._ema: Optional[float] = None
        self._enhance_cfg: EnhanceConfig = config.enhance
        self._enhancer: Optional[AudioEnhancer] = None

        self._mel_params = LogMelSpecParams(
            sample_rate=self._target_sr,
            n_fft=int(config.model.n_fft),
            hop_length=int(config.model.hop_length),
            win_length=int(config.model.win_length),
            n_mels=int(config.model.n_mels),
            fmin=float(config.model.fmin),
            fmax=float(config.model.fmax),
        )
        self._mel_fb = mel_filterbank(
            sample_rate=self._mel_params.sample_rate,
            n_fft=self._mel_params.n_fft,
            n_mels=self._mel_params.n_mels,
            fmin=self._mel_params.fmin,
            fmax=self._mel_params.fmax,
        ).astype(np.float32, copy=False)

        self._onnx: Optional[OnnxModel] = None
        self._hf: Optional[HfAudioClassifier] = None
        if bool(self._enhance_cfg.enabled):
            self._enhancer = AudioEnhancer(self._enhance_cfg, sample_rate=self._target_sr)

        # 1) Try ONNX
        if self._requested_backend in {"onnx", "auto"}:
            model_path = (self._base_dir / str(config.model.path)).resolve()
            if model_path.exists():
                try:
                    self._onnx = OnnxModel(model_path)
                    self._backend_in_use = "onnx"
                except Exception as exc:
                    self._onnx = None
                    self._backend_in_use = "heuristic"
                    self._backend_note = f"ONNX backend unavailable: {exc}"
            else:
                if self._requested_backend == "onnx":
                    self._backend_note = f"ONNX model not found: {model_path}"

        # 2) Try HF (only if requested explicitly OR local snapshot exists in auto mode)
        if self._backend_in_use != "onnx":
            hf_repo_id = str(getattr(config.model, "hf_repo_id", "") or "")
            hf_local_dir = str(getattr(config.model, "hf_local_dir", "") or "")
            hf_revision = str(getattr(config.model, "hf_revision", "") or "")
            local_path = (self._base_dir / hf_local_dir).resolve() if hf_local_dir else None

            allow_hf = self._requested_backend == "hf"
            if self._requested_backend == "auto" and local_path is not None and local_path.exists():
                allow_hf = True

            if allow_hf:
                try:
                    if local_path is not None and local_path.exists():
                        self._hf = HfAudioClassifier(
                            repo_id=hf_repo_id or local_path.as_posix(),
                            local_dir=local_path,
                            revision=hf_revision,
                        )
                    else:
                        if not hf_repo_id:
                            raise RuntimeError("HF backend requires model.hf_repo_id or model.hf_local_dir")
                        self._hf = HfAudioClassifier(
                            repo_id=hf_repo_id,
                            local_dir=None,
                            revision=hf_revision,
                        )
                    self._backend_in_use = "hf"
                except Exception as exc:
                    self._hf = None
                    self._backend_in_use = "heuristic"
                    self._backend_note = f"HF backend unavailable: {exc}"

        if self._backend_in_use == "heuristic":
            if self._requested_backend not in {"auto", "heuristic", "onnx", "hf"}:
                self._backend_note = f"Unknown backend '{self._requested_backend}', using heuristic."
            elif self._backend_note:
                pass
            elif self._requested_backend == "auto":
                self._backend_note = (
                    "ML‑модель не подключена. Для максимальной точности скачайте HF‑модель (scripts/download_hf_model.py) "
                    "или положите ONNX в models/voiceguard.onnx."
                )
            elif self._requested_backend == "heuristic":
                self._backend_note = "Демо‑режим (эвристика). Для максимальной точности подключите ML‑модель (HF/ONNX)."

    @property
    def backend(self) -> str:
        return str(self._backend_in_use)

    @property
    def requested_backend(self) -> str:
        return str(self._requested_backend)

    @property
    def backend_note(self) -> str:
        return str(self._backend_note)

    def reset_state(self) -> None:
        self._ema = None
        if self._enhancer is not None:
            self._enhancer.reset()

    def infer_window(self, audio: np.ndarray, *, orig_sr: int) -> InferenceResult:
        # VAD should see non-normalized signal (dBFS comparable across windows).
        audio_rs = preprocess_audio(audio, orig_sr=orig_sr, target_sr=self._target_sr, normalize=False)
        raw_indicators = extract_indicators(audio_rs, sample_rate=self._target_sr)
        rms_db = float(raw_indicators.get("rms_db", float("-inf")))
        is_speech = bool(rms_db > float(self._config.vad.rms_db_threshold))
        if not is_speech:
            if self._enhancer is not None:
                self._enhancer.update_noise(audio_rs)
            return InferenceResult(
                p_fake=float("nan"),
                p_fake_smooth=float("nan"),
                confidence=0.0,
                is_speech=False,
                indicators=raw_indicators,
                reasons=[],
            )

        audio_proc = audio_rs
        if self._enhancer is not None:
            audio_proc = self._enhancer.process(audio_rs)

        indicators = extract_indicators(audio_proc, sample_rate=self._target_sr)

        backend = self.backend
        reasons: list[str]
        if backend == "onnx":
            if self._onnx is None:  # pragma: no cover
                raise RuntimeError("model.backend=onnx but ONNX model is not initialized.")
            audio_norm = normalize_audio(audio_proc)
            log_mel = log_mel_spectrogram_with_filterbank(
                audio_norm,
                self._mel_params,
                filterbank=self._mel_fb,
            )
            p_fake = float(self._onnx.predict(log_mel))
            reasons = heuristic_reasons(indicators)
            model_confidence = float(abs(p_fake - 0.5) * 2.0)
        elif backend == "hf":
            if self._hf is None:  # pragma: no cover
                raise RuntimeError("model.backend=hf but HF model is not initialized.")
            pred = self._hf.predict(audio_proc, sample_rate=self._target_sr)
            p_fake = float(pred.p_fake)
            reasons = heuristic_reasons(indicators)
            model_confidence = float(pred.model_confidence)
        else:
            p_fake, reasons = heuristic_p_fake(indicators)
            model_confidence = float(abs(p_fake - 0.5) * 2.0)

        if self._ema is None:
            self._ema = float(p_fake)
        else:
            self._ema = float(self._ema_alpha) * float(p_fake) + (1.0 - float(self._ema_alpha)) * float(self._ema)

        p_smooth = float(self._ema)

        # Confidence: blend signal quality + model confidence.
        quality = _clamp((rms_db - float(self._config.vad.rms_db_threshold)) / 30.0, 0.0, 1.0)
        conf = _clamp(float(model_confidence), 0.0, 1.0)
        confidence = _clamp(0.15 + 0.85 * (0.60 * quality + 0.40 * conf), 0.0, 1.0)

        return InferenceResult(
            p_fake=float(_clamp(p_fake, 0.0, 1.0)),
            p_fake_smooth=float(_clamp(p_smooth, 0.0, 1.0)),
            confidence=float(confidence),
            is_speech=True,
            indicators=indicators,
            reasons=reasons,
        )
