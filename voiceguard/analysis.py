from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from voiceguard.alerts import AlertSegment, AlertTracker
from voiceguard.config import AppConfig
from voiceguard.engine import VoiceGuardEngine
from voiceguard.preprocess import preprocess_audio
from voiceguard.types import WindowScore
from voiceguard.windowing import estimate_num_windows, iter_windows, window_params


@dataclass(frozen=True)
class AnalysisSummary:
    total_windows: int
    speech_windows: int
    duration_sec: float
    p_fake_overall: float
    p_fake_median: float
    p_fake_p95: float
    p_fake_mean: float
    p_fake_max: float
    fake_fraction: float
    confidence_mean: float
    confidence_min: float
    alert_segments: list[AlertSegment]


@dataclass(frozen=True)
class AnalysisResult:
    source_kind: str  # file | mic
    source: str
    created_at: str
    sample_rate: int
    window_sec: float
    hop_sec: float
    backend: str
    backend_note: str
    windows: list[WindowScore]
    summary: AnalysisSummary


ProgressCallback = Callable[[int, int], None]


def analyze_audio(
    *,
    audio: np.ndarray,
    orig_sr: int,
    config: AppConfig,
    source_kind: str,
    source: str,
    base_dir: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
) -> AnalysisResult:
    created_at = datetime.now(timezone.utc).isoformat()

    target_sr = int(config.sample_rate)
    audio_rs = preprocess_audio(audio, orig_sr=orig_sr, target_sr=target_sr, normalize=False)

    window_samples, hop_samples = window_params(
        sample_rate=target_sr, window_sec=float(config.window_sec), hop_sec=float(config.hop_sec)
    )
    total = estimate_num_windows(int(audio_rs.size), window_samples=window_samples, hop_samples=hop_samples)

    engine = VoiceGuardEngine(config, base_dir=base_dir)
    engine.reset_state()

    hold_sec = float(config.alert_hold_sec)
    if str(source_kind).lower() == "file":
        # In offline mode we prefer to highlight suspicious parts even if they are short.
        # Trigger on a single step to avoid missing brief synthetic segments.
        hold_sec = min(hold_sec, float(config.hop_sec))

    alert = AlertTracker(
        threshold=float(config.alert_threshold),
        hold_sec=float(hold_sec),
        step_sec=float(config.hop_sec),
    )

    windows: list[WindowScore] = []
    use_raw_for_alert = str(source_kind).lower() == "file"
    processed = 0
    for start, win in iter_windows(audio_rs, window_samples=window_samples, hop_samples=hop_samples):
        t_start = float(start) / float(target_sr)
        t_end = float(start + window_samples) / float(target_sr)
        result = engine.infer_window(win, orig_sr=target_sr)
        windows.append(WindowScore(t_start=t_start, t_end=t_end, result=result))

        if result.is_speech:
            p_alert = float(result.p_fake if use_raw_for_alert else result.p_fake_smooth)
            alert.update(t_start=t_start, t_end=t_end, p=p_alert, is_speech=True)
        else:
            alert.update(t_start=t_start, t_end=t_end, p=0.0, is_speech=False)

        processed += 1
        if progress is not None and (processed == total or processed % 10 == 0):
            progress(processed, total)

    if windows:
        alert.finalize(t_end=float(windows[-1].t_end))

    # For offline scoring prefer raw model probability (EMA smoothing is mainly for live UI stability).
    p_vals = [float(w.result.p_fake) for w in windows if w.result.is_speech and not np.isnan(float(w.result.p_fake))]
    conf_vals = [float(w.result.confidence) for w in windows if w.result.is_speech]
    speech_windows = int(len(p_vals))
    p_mean = float(np.mean(p_vals)) if p_vals else 0.0
    p_max = float(np.max(p_vals)) if p_vals else 0.0
    p_median = float(np.median(p_vals)) if p_vals else 0.0
    p_p95 = float(np.quantile(p_vals, 0.95)) if p_vals else 0.0
    fake_fraction = float(np.mean(np.asarray(p_vals) >= float(config.alert_threshold))) if p_vals else 0.0
    conf_mean = float(np.mean(conf_vals)) if conf_vals else 0.0
    conf_min = float(np.min(conf_vals)) if conf_vals else 0.0
    duration_sec = float(audio_rs.size) / float(target_sr) if target_sr > 0 else 0.0

    # Extra "whole clip" inference (HF models often work better on longer context than 1–2s windows).
    p_clip = p_p95
    if str(engine.backend).lower() == "hf" and p_vals:
        try:
            # Pick a short segment starting from the first detected speech window.
            first_speech_start = 0.0
            for w in windows:
                if w.result.is_speech:
                    first_speech_start = float(w.t_start)
                    break
            start = int(first_speech_start * float(target_sr))
            clip_len = min(int(audio_rs.size - start), int(float(target_sr) * 12.0))
            if clip_len > 0:
                clip_audio = audio_rs[start : start + clip_len]
                clip_res = engine.infer_window(clip_audio, orig_sr=target_sr)
                if clip_res.is_speech and not np.isnan(float(clip_res.p_fake)):
                    p_clip = float(clip_res.p_fake)
        except Exception:
            p_clip = p_p95

    p_overall = float(max(float(p_p95), float(p_clip)))

    summary = AnalysisSummary(
        total_windows=int(total),
        speech_windows=speech_windows,
        duration_sec=duration_sec,
        # Final score used for the big verdict number in UI (prefer whole-clip HF inference when available).
        p_fake_overall=float(p_overall),
        p_fake_median=float(p_median),
        p_fake_p95=float(p_p95),
        p_fake_mean=float(p_mean),
        p_fake_max=float(p_max),
        fake_fraction=float(fake_fraction),
        confidence_mean=float(conf_mean),
        confidence_min=float(conf_min),
        alert_segments=alert.segments,
    )

    return AnalysisResult(
        source_kind=str(source_kind),
        source=str(source),
        created_at=created_at,
        sample_rate=target_sr,
        window_sec=float(config.window_sec),
        hop_sec=float(config.hop_sec),
        backend=str(engine.backend),
        backend_note=str(engine.backend_note),
        windows=windows,
        summary=summary,
    )
