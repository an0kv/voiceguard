from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "auto"  # auto | heuristic | onnx | hf
    path: str = "models/voiceguard.onnx"
    hf_repo_id: str = ""
    hf_local_dir: str = "models/hf_model"
    hf_revision: str = ""
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    fmin: int = 20
    fmax: int = 7600


@dataclass(frozen=True)
class SmoothingConfig:
    ema_alpha: float = 0.35


@dataclass(frozen=True)
class VadConfig:
    rms_db_threshold: float = -45.0


@dataclass(frozen=True)
class EnhanceConfig:
    enabled: bool = False
    bandpass: bool = True
    bandpass_low_hz: float = 80.0
    bandpass_high_hz: float = 8000.0
    noise_reduction: bool = True
    noise_strength: float = 0.80
    noise_ema: float = 0.10


@dataclass(frozen=True)
class StorageConfig:
    store_audio: bool = False
    store_metadata: bool = True
    reports_dir: str = "reports"


@dataclass(frozen=True)
class AppConfig:
    sample_rate: int = 16000
    window_sec: float = 1.0
    hop_sec: float = 0.25
    alert_threshold: float = 0.80
    alert_hold_sec: float = 3.0
    model: ModelConfig = ModelConfig()
    smoothing: SmoothingConfig = SmoothingConfig()
    vad: VadConfig = VadConfig()
    enhance: EnhanceConfig = EnhanceConfig()
    storage: StorageConfig = StorageConfig()


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    value = d.get(key, default)
    return default if value is None else value


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        return AppConfig()

    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read config.yaml") from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return AppConfig()

    model_raw = raw.get("model", {}) or {}
    smoothing_raw = raw.get("smoothing", {}) or {}
    vad_raw = raw.get("vad", {}) or {}
    enhance_raw = raw.get("enhance", {}) or {}
    storage_raw = raw.get("storage", {}) or {}

    return AppConfig(
        sample_rate=int(_get(raw, "sample_rate", 16000)),
        window_sec=float(_get(raw, "window_sec", 1.0)),
        hop_sec=float(_get(raw, "hop_sec", 0.25)),
        alert_threshold=float(_get(raw, "alert_threshold", 0.80)),
        alert_hold_sec=float(_get(raw, "alert_hold_sec", 3.0)),
        model=ModelConfig(
            backend=str(_get(model_raw, "backend", "auto")),
            path=str(_get(model_raw, "path", "models/voiceguard.onnx")),
            hf_repo_id=str(_get(model_raw, "hf_repo_id", "")),
            hf_local_dir=str(_get(model_raw, "hf_local_dir", "models/hf_model")),
            hf_revision=str(_get(model_raw, "hf_revision", "")),
            n_mels=int(_get(model_raw, "n_mels", 64)),
            n_fft=int(_get(model_raw, "n_fft", 512)),
            hop_length=int(_get(model_raw, "hop_length", 160)),
            win_length=int(_get(model_raw, "win_length", 400)),
            fmin=int(_get(model_raw, "fmin", 20)),
            fmax=int(_get(model_raw, "fmax", 7600)),
        ),
        smoothing=SmoothingConfig(
            ema_alpha=float(_get(smoothing_raw, "ema_alpha", 0.35)),
        ),
        vad=VadConfig(
            rms_db_threshold=float(_get(vad_raw, "rms_db_threshold", -45.0)),
        ),
        enhance=EnhanceConfig(
            enabled=bool(_get(enhance_raw, "enabled", False)),
            bandpass=bool(_get(enhance_raw, "bandpass", True)),
            bandpass_low_hz=float(_get(enhance_raw, "bandpass_low_hz", 80.0)),
            bandpass_high_hz=float(_get(enhance_raw, "bandpass_high_hz", 8000.0)),
            noise_reduction=bool(_get(enhance_raw, "noise_reduction", True)),
            noise_strength=float(_get(enhance_raw, "noise_strength", 0.80)),
            noise_ema=float(_get(enhance_raw, "noise_ema", 0.10)),
        ),
        storage=StorageConfig(
            store_audio=bool(_get(storage_raw, "store_audio", False)),
            store_metadata=bool(_get(storage_raw, "store_metadata", True)),
            reports_dir=str(_get(storage_raw, "reports_dir", "reports")),
        ),
    )
