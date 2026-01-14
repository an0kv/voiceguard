from __future__ import annotations

import queue
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QGridLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from voiceguard.alerts import AlertTracker
from voiceguard.audio.mic_capture import MicCapture
from voiceguard.audio.system_capture import SystemAudioCapture
from voiceguard.config import AppConfig, EnhanceConfig
from voiceguard.dsp.resample import resample_audio
from voiceguard.engine import VoiceGuardEngine
from voiceguard.types import InferenceResult
from voiceguard.ui.presentation import format_percent, make_verdict
from voiceguard.ui.widgets.timeline import TimeSegment, TimelineWidget
from voiceguard.windowing import StreamWindowProcessor


@dataclass(frozen=True)
class _DeviceSpec:
    device: Optional[int]
    loopback: bool


class _LiveAnalyzerThread(QThread):
    point = Signal(object)  # dict payload
    error = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        *,
        config: AppConfig,
        source: str,
        device: Optional[int],
        loopback: bool,
    ) -> None:
        super().__init__()
        self._config = config
        self._source = str(source)
        self._device = device
        self._loopback = bool(loopback)

    def run(self) -> None:  # noqa: PLR0912
        target_sr = int(self._config.sample_rate)
        if self._source == "system":
            capture = SystemAudioCapture(device=self._device, loopback=self._loopback)
            source_label = "системы"
        else:
            capture = MicCapture(device=self._device)
            source_label = "микрофона"

        try:
            actual_sr = int(capture.start(preferred_sample_rate=target_sr, block_sec=0.10))
        except Exception as exc:
            self.error.emit(str(exc))
            return

        resample_needed = int(actual_sr) != int(target_sr)
        if resample_needed:
            sr_status = f"SR {source_label}: {actual_sr} Hz → ресемплинг до {target_sr} Hz"
        else:
            sr_status = f"SR {source_label}: {actual_sr} Hz"

        processor = StreamWindowProcessor(
            sample_rate=target_sr,
            window_sec=float(self._config.window_sec),
            hop_sec=float(self._config.hop_sec),
        )
        engine = VoiceGuardEngine(self._config)
        engine.reset_state()

        backend = str(engine.backend).lower()
        if backend == "onnx":
            backend_status = "Режим: ML модель (ONNX)"
        elif backend == "hf":
            backend_status = "Режим: ML модель (HuggingFace)"
        else:
            backend_status = str(engine.backend_note) or "Режим: Демо (эвристика). Для максимальной точности подключите ML‑модель (HF/ONNX)."

        if self._source == "system":
            if isinstance(capture, SystemAudioCapture) and capture.loopback_active:
                source_status = "Источник: системный звук (WASAPI loopback)"
            elif self._loopback:
                source_status = "Источник: системный звук (виртуальное устройство)"
            else:
                source_status = "Источник: системный звук"
        else:
            source_status = "Источник: микрофон"

        enhance_status = ""
        if bool(getattr(self._config, "enhance", None) and self._config.enhance.enabled):
            parts = []
            if self._config.enhance.bandpass:
                parts.append("фильтр речи")
            if self._config.enhance.noise_reduction:
                parts.append("шумоподавление")
            if parts:
                enhance_status = " • Фокус: " + "+".join(parts)

        self.status.emit(f"{source_status} • {sr_status} • {backend_status}{enhance_status}")

        alert = AlertTracker(
            threshold=float(self._config.alert_threshold),
            hold_sec=float(self._config.alert_hold_sec),
            step_sec=float(self._config.hop_sec),
        )

        try:
            while not self.isInterruptionRequested():
                try:
                    chunk = capture.queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                samples = chunk.samples.astype(np.float32, copy=False)
                if resample_needed and int(chunk.sample_rate) != int(target_sr):
                    samples = resample_audio(samples, orig_sr=int(chunk.sample_rate), target_sr=target_sr)

                for w in processor.push(samples):
                    t_start = float(w.start_sample) / float(target_sr)
                    t_end = float(w.start_sample + processor.window_samples) / float(target_sr)

                    result = engine.infer_window(w.samples, orig_sr=target_sr)
                    if result.is_speech:
                        is_alert = alert.update(
                            t_start=t_start,
                            t_end=t_end,
                            p=float(result.p_fake_smooth),
                            is_speech=True,
                        )
                    else:
                        is_alert = alert.update(t_start=t_start, t_end=t_end, p=0.0, is_speech=False)

                    self.point.emit(
                        {
                            "t_start": t_start,
                            "t_end": t_end,
                            "result": result,
                            "alert": bool(is_alert),
                            "segments": alert.segments,
                        }
                    )
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            capture.stop()


class LiveTab(QWidget):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._thread: Optional[_LiveAnalyzerThread] = None

        self._times: "deque[float]" = deque(maxlen=400)  # ~100s at 0.25s hop
        self._values: "deque[Optional[float]]" = deque(maxlen=400)
        self._segments: list[TimeSegment] = []
        self._noise_floor_db: Optional[float] = None
        self._applying_profile = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        controls = QVBoxLayout()
        controls.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(8)
        self._source_combo = QComboBox()
        self._source_combo.addItem("Микрофон", "mic")
        self._source_combo.addItem("Системный звук (Zoom/Discord)", "system")
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._device_combo = QComboBox()
        self._device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._start_btn = QPushButton("Старт")
        self._start_btn.clicked.connect(self._toggle)
        refresh_btn = QPushButton("Обновить")
        refresh_btn.setProperty("secondary", True)
        refresh_btn.clicked.connect(self._refresh_devices)

        top.addWidget(QLabel("Источник:"))
        top.addWidget(self._source_combo)
        top.addWidget(QLabel("Устройство:"))
        top.addWidget(self._device_combo, 1)
        top.addWidget(refresh_btn)
        top.addWidget(self._start_btn)
        controls.addLayout(top)

        self._profile_combo = QComboBox()
        self._profile_combo.addItem("Пользовательский", "custom")
        self._profile_combo.addItem("Созвон (Zoom/Discord)", "call")
        self._profile_combo.addItem("Конференция (шумная среда)", "noisy")
        self._profile_combo.addItem("Студия/подкаст", "studio")
        self._profile_combo.currentIndexChanged.connect(self._on_profile_changed)

        profile_row = QHBoxLayout()
        profile_row.setSpacing(8)
        profile_row.addWidget(QLabel("Профиль:"))
        profile_row.addWidget(self._profile_combo, 1)
        controls.addLayout(profile_row)

        layout.addLayout(controls)

        self._source_hint = QLabel()
        self._source_hint.setWordWrap(True)
        self._source_hint.setProperty("muted", True)
        layout.addWidget(self._source_hint)

        header = QFrame()
        header.setProperty("card", True)
        header_layout = QVBoxLayout(header)
        header_layout.setSpacing(6)
        self._header_title = QLabel("Онлайн анализ голоса")
        self._header_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        header_layout.addWidget(self._header_title)

        chip_row = QHBoxLayout()
        self._chip_source = QLabel("Источник: —")
        self._chip_source.setProperty("chip", True)
        self._chip_profile = QLabel("Профиль: пользовательский")
        self._chip_profile.setProperty("chip", True)
        self._chip_backend = QLabel("Режим: —")
        self._chip_backend.setProperty("chip", True)
        self._chip_focus = QLabel("Фокус: выкл")
        self._chip_focus.setProperty("chip", True)
        for chip in (self._chip_source, self._chip_profile, self._chip_backend, self._chip_focus):
            chip_row.addWidget(chip)
        chip_row.addStretch(1)
        header_layout.addLayout(chip_row)

        self._prob_big = QLabel("—")
        self._prob_big.setProperty("tone", "hero")
        self._set_prob_style()
        header_layout.addWidget(self._prob_big)

        self._verdict_title = QLabel("Нажмите «Старт»")
        self._verdict_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self._verdict_subtitle = QLabel(
            "VoiceGuard оценивает вероятность того, что голос сгенерирован ИИ (TTS/voice clone)."
        )
        self._verdict_subtitle.setWordWrap(True)
        self._verdict_subtitle.setProperty("muted", True)
        header_layout.addWidget(self._verdict_title)
        header_layout.addWidget(self._verdict_subtitle)

        stats_grid = QGridLayout()

        def _stat_card(title: str) -> tuple[QFrame, QLabel]:
            card = QFrame()
            card.setProperty("stat", True)
            v = QVBoxLayout(card)
            label = QLabel(title)
            label.setProperty("muted", True)
            label.setStyleSheet("font-size: 11px;")
            value = QLabel("—")
            value.setStyleSheet("font-size: 16px; font-weight: 700;")
            v.addWidget(label)
            v.addWidget(value)
            return card, value

        avg_card, self._avg_label = _stat_card("Среднее 30с")
        peak_card, self._peak_label = _stat_card("Пик")
        signal_card, self._signal_label = _stat_card("Сигнал (dB)")
        noise_card, self._noise_label = _stat_card("Шум (dB)")
        snr_card, self._snr_label = _stat_card("SNR")
        speech_card, self._speech_label = _stat_card("Речь")

        stats_grid.addWidget(avg_card, 0, 0)
        stats_grid.addWidget(peak_card, 0, 1)
        stats_grid.addWidget(signal_card, 0, 2)
        stats_grid.addWidget(noise_card, 1, 0)
        stats_grid.addWidget(snr_card, 1, 1)
        stats_grid.addWidget(speech_card, 1, 2)
        header_layout.addLayout(stats_grid)

        layout.addWidget(header)

        focus_box = QGroupBox("Фокус на речи")
        focus_layout = QFormLayout(focus_box)
        self._enhance_toggle = QCheckBox("Включить улучшение сигнала")
        self._bandpass_toggle = QCheckBox("Фильтр речи 80–8000 Hz")
        self._noise_toggle = QCheckBox("Авто‑шумоподавление")
        self._bandpass_toggle.setChecked(True)
        self._noise_toggle.setChecked(True)
        self._enhance_toggle.setChecked(bool(self._config.enhance.enabled))
        self._bandpass_toggle.setChecked(bool(self._config.enhance.bandpass))
        self._noise_toggle.setChecked(bool(self._config.enhance.noise_reduction))
        self._enhance_toggle.stateChanged.connect(self._sync_enhance_controls)
        self._enhance_toggle.stateChanged.connect(self._mark_custom_profile)
        self._bandpass_toggle.stateChanged.connect(self._mark_custom_profile)
        self._noise_toggle.stateChanged.connect(self._mark_custom_profile)

        self._enhance_hint = QLabel(
            "Совет: включайте фокус для Zoom/Discord/Meet, чтобы уменьшить влияние шума и фоновой музыки."
        )
        self._enhance_hint.setWordWrap(True)
        self._enhance_hint.setProperty("muted", True)
        self._routing_btn = QPushButton("Как изолировать звук Zoom/Discord")
        self._routing_btn.setProperty("secondary", True)
        self._routing_btn.clicked.connect(self._show_routing_help)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.50, 0.99)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(float(self._config.alert_threshold))
        self._threshold_spin.valueChanged.connect(self._mark_custom_profile)

        focus_layout.addRow(self._enhance_toggle)
        focus_layout.addRow(self._bandpass_toggle)
        focus_layout.addRow(self._noise_toggle)
        focus_layout.addRow(self._enhance_hint)
        focus_layout.addRow(self._routing_btn)
        focus_layout.addRow("Порог алерта:", self._threshold_spin)
        layout.addWidget(focus_box)

        stats_box = QGroupBox("Детали")
        stats = QFormLayout(stats_box)
        self._conf_label = QLabel("—")
        self._alert_label = QLabel("—")
        self._status_label = QLabel("—")
        self._status_label.setWordWrap(True)
        stats.addRow("Уверенность:", self._conf_label)
        stats.addRow("Предупреждение:", self._alert_label)
        stats.addRow("Статус:", self._status_label)
        stats.addRow("Окно/шаг:", QLabel(f"{self._config.window_sec:.2f}s / {self._config.hop_sec:.2f}s"))
        layout.addWidget(stats_box)

        self._timeline = TimelineWidget()
        self._timeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._timeline, 1)

        layout.addWidget(QLabel("Почему так (простые индикаторы):"))
        self._reasons = QListWidget()
        layout.addWidget(self._reasons)

        self._note_label = QLabel()
        self._note_label.setWordWrap(True)
        self._note_label.setProperty("muted", True)
        layout.addWidget(self._note_label)

        self._loopback_available = False
        self._refresh_devices()
        self._sync_enhance_controls()
        self._update_focus_chip()
        self._update_source_texts()

    def _refresh_devices(self) -> None:
        self._device_combo.clear()
        source = self._current_source()
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            self._device_combo.addItem("sounddevice недоступен", None)
            self._device_combo.setEnabled(False)
            self._start_btn.setEnabled(False)
            return

        self._device_combo.setEnabled(True)
        self._start_btn.setEnabled(True)

        try:
            hostapis = sd.query_hostapis()
        except Exception:
            hostapis = []

        hostapi_names = {idx: str(api.get("name", "")) for idx, api in enumerate(hostapis)}
        has_wasapi = bool(
            hasattr(sd, "WasapiSettings")
            and any("WASAPI" in name.upper() for name in hostapi_names.values())
        )
        self._loopback_available = bool(source == "system" and has_wasapi)

        default_loopback = bool(source == "system" and has_wasapi)
        self._device_combo.addItem("По умолчанию", _DeviceSpec(device=None, loopback=default_loopback))

        try:
            devices = sd.query_devices()
        except Exception:
            return

        for idx, dev in enumerate(devices):
            try:
                if source == "mic":
                    if int(dev.get("max_input_channels", 0)) <= 0:
                        continue
                    name = str(dev.get("name", f"Device {idx}"))
                    self._device_combo.addItem(f"{idx}: {name}", _DeviceSpec(device=int(idx), loopback=False))
                else:
                    if has_wasapi:
                        if int(dev.get("max_output_channels", 0)) <= 0:
                            continue
                        hostapi_name = hostapi_names.get(int(dev.get("hostapi", -1)), "")
                        if "WASAPI" not in str(hostapi_name).upper():
                            continue
                        name = str(dev.get("name", f"Device {idx}"))
                        self._device_combo.addItem(f"{idx}: {name}", _DeviceSpec(device=int(idx), loopback=True))
                    else:
                        if int(dev.get("max_input_channels", 0)) <= 0:
                            continue
                        name = str(dev.get("name", f"Device {idx}"))
                        self._device_combo.addItem(f"{idx}: {name}", _DeviceSpec(device=int(idx), loopback=False))
            except Exception:
                continue

    def _current_source(self) -> str:
        data = self._source_combo.currentData()
        return str(data) if data else "mic"

    def _on_profile_changed(self, *_: object) -> None:
        profile = self._profile_combo.currentData()
        if profile == "custom":
            self._chip_profile.setText("Профиль: пользовательский")
            self._update_focus_chip()
            return

        self._applying_profile = True
        if profile == "call":
            self._enhance_toggle.setChecked(True)
            self._bandpass_toggle.setChecked(True)
            self._noise_toggle.setChecked(True)
            self._threshold_spin.setValue(0.78)
            self._chip_profile.setText("Профиль: созвон")
        elif profile == "noisy":
            self._enhance_toggle.setChecked(True)
            self._bandpass_toggle.setChecked(True)
            self._noise_toggle.setChecked(True)
            self._threshold_spin.setValue(0.82)
            self._chip_profile.setText("Профиль: шумная среда")
        elif profile == "studio":
            self._enhance_toggle.setChecked(False)
            self._bandpass_toggle.setChecked(False)
            self._noise_toggle.setChecked(False)
            self._threshold_spin.setValue(0.85)
            self._chip_profile.setText("Профиль: студия")

        self._applying_profile = False
        self._sync_enhance_controls()
        self._update_focus_chip()

    def _sync_enhance_controls(self, *_: object) -> None:
        enabled = bool(self._enhance_toggle.isChecked())
        self._bandpass_toggle.setEnabled(enabled)
        self._noise_toggle.setEnabled(enabled)

    def _mark_custom_profile(self, *_: object) -> None:
        if self._applying_profile:
            return
        if self._profile_combo.currentData() != "custom":
            self._profile_combo.setCurrentIndex(0)
            self._chip_profile.setText("Профиль: пользовательский")
        self._update_focus_chip()

    def _build_live_config(self) -> AppConfig:
        threshold = float(self._threshold_spin.value())
        base = self._config.enhance
        enhance = EnhanceConfig(
            enabled=bool(self._enhance_toggle.isChecked()),
            bandpass=bool(self._bandpass_toggle.isChecked()),
            bandpass_low_hz=float(base.bandpass_low_hz),
            bandpass_high_hz=float(base.bandpass_high_hz),
            noise_reduction=bool(self._noise_toggle.isChecked()),
            noise_strength=float(base.noise_strength),
            noise_ema=float(base.noise_ema),
        )
        return replace(self._config, alert_threshold=threshold, enhance=enhance)

    def _selected_device(self) -> _DeviceSpec:
        data = self._device_combo.currentData()
        if isinstance(data, _DeviceSpec):
            return data
        if isinstance(data, int):
            return _DeviceSpec(device=int(data), loopback=False)
        return _DeviceSpec(device=None, loopback=False)

    def _on_source_changed(self, *_: object) -> None:
        self._refresh_devices()
        self._update_source_texts()

    def _update_source_texts(self) -> None:
        source = self._current_source()
        if source == "system":
            self._header_title.setText("Онлайн анализ голоса (системный звук)")
            self._verdict_title.setText("Нажмите «Старт» и включите звонок/конференцию")
            self._verdict_subtitle.setText(
                "VoiceGuard анализирует системный звук (Zoom/Discord/Meet) в реальном времени."
            )
            self._chip_source.setText("Источник: системный звук")
            if self._loopback_available:
                self._source_hint.setText(
                    "Windows: используется WASAPI loopback для захвата системного звука. "
                    "Чтобы слушать только Zoom/Discord, направьте их вывод на отдельное устройство."
                )
            else:
                self._source_hint.setText(
                    "macOS/Linux: нужен виртуальный loopback (BlackHole/Soundflower/PulseAudio Monitor). "
                    "Для изоляции приложений направьте Zoom/Discord на отдельный виртуальный выход."
                )
            self._note_label.setText(
                "Важно: это вероятностная оценка. На точность влияют кодеки VoIP/пересжатие, шум и качество звука."
            )
        else:
            self._header_title.setText("Онлайн анализ голоса (микрофон)")
            self._verdict_title.setText("Нажмите «Старт» и говорите в микрофон")
            self._verdict_subtitle.setText(
                "VoiceGuard оценивает вероятность того, что голос сгенерирован ИИ (TTS/voice clone)."
            )
            self._chip_source.setText("Источник: микрофон")
            self._source_hint.setText("Совет: для Zoom/Discord выберите «Системный звук».")
            self._note_label.setText(
                "Важно: это вероятностная оценка. На точность влияют шум, пересжатие/VoIP и качество микрофона."
            )

    def _update_focus_chip(self) -> None:
        if not self._enhance_toggle.isChecked():
            self._chip_focus.setText("Фокус: выкл")
            return
        parts = []
        if self._bandpass_toggle.isChecked():
            parts.append("фильтр")
        if self._noise_toggle.isChecked():
            parts.append("шум")
        label = "+".join(parts) if parts else "on"
        self._chip_focus.setText(f"Фокус: {label}")

    def _set_prob_style(self, color: Optional[str] = None) -> None:
        base = "font-size: 42px; font-weight: 800;"
        if color:
            self._prob_big.setStyleSheet(f"{base} color: {color};")
        else:
            self._prob_big.setStyleSheet(base)

    def _toggle(self) -> None:
        if self._thread is None:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        if self._thread is not None:
            return
        source = self._current_source()
        device_spec = self._selected_device()
        live_config = self._build_live_config()

        self._times.clear()
        self._values.clear()
        self._segments = []
        self._timeline.clear()
        self._reasons.clear()
        self._noise_floor_db = None

        self._thread = _LiveAnalyzerThread(
            config=live_config,
            source=source,
            device=device_spec.device,
            loopback=device_spec.loopback,
        )
        self._thread.point.connect(self._on_point)
        self._thread.error.connect(self._on_error)
        self._thread.status.connect(self._on_status)
        self._thread.finished.connect(self._on_stopped)
        self._start_btn.setText("Стоп")
        self._source_combo.setEnabled(False)
        self._device_combo.setEnabled(False)
        self._profile_combo.setEnabled(False)
        self._enhance_toggle.setEnabled(False)
        self._bandpass_toggle.setEnabled(False)
        self._noise_toggle.setEnabled(False)
        self._threshold_spin.setEnabled(False)
        self._thread.start()

    def _stop(self) -> None:
        if self._thread is None:
            return
        self._start_btn.setEnabled(False)
        self._thread.requestInterruption()
        self._thread.wait(1500)
        # If it still didn't stop, let Qt finish it; UI state updates in _on_stopped.

    def _on_stopped(self) -> None:
        self._start_btn.setEnabled(True)
        self._start_btn.setText("Старт")
        self._source_combo.setEnabled(True)
        self._device_combo.setEnabled(True)
        self._profile_combo.setEnabled(True)
        self._enhance_toggle.setEnabled(True)
        self._threshold_spin.setEnabled(True)
        self._sync_enhance_controls()
        self._thread = None
        self._alert_label.setText("—")
        self._avg_label.setText("—")
        self._peak_label.setText("—")
        self._signal_label.setText("—")
        self._noise_label.setText("—")
        self._snr_label.setText("—")
        self._speech_label.setText("—")

    def _on_status(self, text: str) -> None:
        self._status_label.setText(text)
        backend = "—"
        if "ONNX" in text:
            backend = "ONNX"
        elif "HuggingFace" in text:
            backend = "HF"
        elif "Демо" in text or "эвристика" in text:
            backend = "Heuristic"
        self._chip_backend.setText(f"Режим: {backend}")

    def _show_routing_help(self) -> None:
        QMessageBox.information(
            self,
            "Изоляция звука приложений",
            "Windows: откройте «Параметры звука → Дополнительные параметры громкости приложений» "
            "и направьте Zoom/Discord на отдельное устройство вывода.\n\n"
            "macOS: используйте BlackHole/Loopback и задайте вывод Zoom/Discord на виртуальное устройство.\n\n"
            "Linux: PulseAudio/PIPEWIRE — выберите Monitor‑устройство для приложения.",
        )

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "VoiceGuard", message or "Неизвестная ошибка")
        self._on_stopped()

    def _on_point(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        result = payload.get("result")
        if not isinstance(result, InferenceResult):
            return

        t_end = float(payload.get("t_end", 0.0))
        self._times.append(t_end)
        v = float(result.p_fake_smooth)
        self._values.append(None if (not result.is_speech or v != v) else v)

        # Update segments (last state).
        segs = payload.get("segments", [])
        self._segments = [TimeSegment(start_sec=s.start_sec, end_sec=s.end_sec) for s in segs] if segs else []

        self._timeline.set_data(
            times=list(self._times),
            values=list(self._values),
            threshold=float(self._threshold_spin.value()),
            segments=self._segments,
        )

        window_sec = 30.0
        recent_vals = [
            float(val)
            for t, val in zip(self._times, self._values)
            if val is not None and float(t) >= float(t_end - window_sec)
        ]
        if recent_vals:
            avg = float(sum(recent_vals)) / float(len(recent_vals))
            peak = float(max(recent_vals))
            self._avg_label.setText(format_percent(avg))
            self._peak_label.setText(format_percent(peak))
        else:
            self._avg_label.setText("—")
            self._peak_label.setText("—")

        rms_db = float(result.indicators.get("rms_db", float("nan")))
        if not result.is_speech and rms_db == rms_db:
            if self._noise_floor_db is None:
                self._noise_floor_db = float(rms_db)
            else:
                self._noise_floor_db = 0.90 * float(self._noise_floor_db) + 0.10 * float(rms_db)

        if rms_db == rms_db:
            self._signal_label.setText(f"{rms_db:.0f} dB")
        else:
            self._signal_label.setText("—")

        if self._noise_floor_db is not None:
            self._noise_label.setText(f"{self._noise_floor_db:.0f} dB")
            if rms_db == rms_db:
                snr = float(rms_db) - float(self._noise_floor_db)
                self._snr_label.setText(f"{snr:.0f} dB")
            else:
                self._snr_label.setText("—")
        else:
            self._noise_label.setText("—")
            self._snr_label.setText("—")

        self._speech_label.setText("да" if result.is_speech else "нет")

        if not result.is_speech:
            self._prob_big.setText("—")
            self._set_prob_style()
            self._verdict_title.setText("Тишина / нет речи")
            self._verdict_subtitle.setText("Говорите ближе к микрофону или увеличьте громкость входа.")
            self._conf_label.setText(format_percent(0.0))
            self._alert_label.setText("—")
        else:
            p = float(result.p_fake_smooth)
            verdict = make_verdict(
                p,
                confidence=float(result.confidence),
                threshold=float(self._threshold_spin.value()),
            )
            self._prob_big.setText(format_percent(p))
            self._set_prob_style(verdict.color)
            self._verdict_title.setText(verdict.title)
            self._verdict_subtitle.setText(verdict.subtitle)

            self._conf_label.setText(format_percent(float(result.confidence)))
            is_alert = bool(payload.get("alert", False))
            self._alert_label.setText("АКТИВНО" if is_alert else "—")

        self._reasons.clear()
        if result.reasons:
            for r in result.reasons[:6]:
                self._reasons.addItem(str(r))
        else:
            self._reasons.addItem("—")
