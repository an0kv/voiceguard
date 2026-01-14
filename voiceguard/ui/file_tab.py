from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QFrame,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from voiceguard.config import AppConfig
from voiceguard.analysis import AnalysisResult, analyze_audio
from voiceguard.audio.file_reader import load_audio_file
from voiceguard.reports import (
    default_report_stem,
    ensure_reports_dir,
    write_html_report,
    write_json_report,
)
from voiceguard.ui.presentation import Verdict, format_percent, make_verdict
from voiceguard.ui.widgets.timeline import TimeSegment, TimelineWidget


class _FileAnalyzeWorker(QObject):
    progress = Signal(int, int)  # done, total
    finished = Signal(object)  # AnalysisResult
    error = Signal(str)

    def __init__(self, *, path: Path, config: AppConfig) -> None:
        super().__init__()
        self._path = Path(path)
        self._config = config

    def run(self) -> None:
        try:
            audio, sr = load_audio_file(self._path)

            def on_progress(done: int, total: int) -> None:
                self.progress.emit(int(done), int(total))

            analysis = analyze_audio(
                audio=audio,
                orig_sr=int(sr),
                config=self._config,
                source_kind="file",
                source=str(self._path),
                progress=on_progress,
            )
            self.finished.emit(analysis)
        except Exception as exc:
            self.error.emit(str(exc))


class FileTab(QWidget):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._analysis: Optional[AnalysisResult] = None

        self._thread: Optional[QThread] = None
        self._worker: Optional[_FileAnalyzeWorker] = None

        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = QFrame()
        header.setProperty("card", True)
        header_layout = QVBoxLayout(header)
        header_layout.setSpacing(6)
        header_title = QLabel("Проверка аудиофайла")
        header_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        header_layout.addWidget(header_title)

        self._prob_big = QLabel("—")
        self._prob_big.setProperty("tone", "hero")
        self._set_prob_style()
        header_layout.addWidget(self._prob_big)

        self._verdict_title = QLabel("Выберите файл для анализа")
        self._verdict_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self._verdict_subtitle = QLabel("Перетащите .wav/.flac/.mp3 или нажмите «Выбрать…»")
        self._verdict_subtitle.setWordWrap(True)
        self._verdict_subtitle.setProperty("muted", True)
        header_layout.addWidget(self._verdict_title)
        header_layout.addWidget(self._verdict_subtitle)

        layout.addWidget(header)

        chips = QHBoxLayout()
        chips.setSpacing(8)
        self._chip_backend = QLabel("Режим: —")
        self._chip_backend.setProperty("chip", True)
        self._chip_duration = QLabel("Длительность: —")
        self._chip_duration.setProperty("chip", True)
        self._chip_speech = QLabel("Речь: —")
        self._chip_speech.setProperty("chip", True)
        self._chip_alerts = QLabel("Алерты: —")
        self._chip_alerts.setProperty("chip", True)
        for chip in (self._chip_backend, self._chip_duration, self._chip_speech, self._chip_alerts):
            chips.addWidget(chip)
        chips.addStretch(1)
        layout.addLayout(chips)

        file_row = QHBoxLayout()
        file_row.setSpacing(10)
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText("Перетащите файл (.wav/.flac/.mp3) сюда или нажмите «Выбрать…»")
        browse_btn = QPushButton("Выбрать…")
        browse_btn.setProperty("secondary", True)
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(self._path_edit, 1)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self._analyze_btn = QPushButton("Проверить")
        self._analyze_btn.clicked.connect(self._start_analysis)
        self._export_btn = QPushButton("Экспорт отчёта…")
        self._export_btn.setProperty("secondary", True)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_report)
        actions.addWidget(self._analyze_btn)
        actions.addWidget(self._export_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        summary = QFormLayout()
        self._backend = QLabel("—")
        self._backend_note = QLabel("")
        self._backend_note.setWordWrap(True)
        self._backend_note.setProperty("muted", True)
        self._p_mean = QLabel("—")
        self._p_max = QLabel("—")
        self._fake_fraction = QLabel("—")
        self._conf_mean = QLabel("—")
        self._speech_windows = QLabel("—")
        summary.addRow("Режим:", self._backend)
        summary.addRow("", self._backend_note)
        summary.addRow("Средняя вероятность ИИ:", self._p_mean)
        summary.addRow("Максимальная вероятность ИИ:", self._p_max)
        summary.addRow("Доля окон выше порога:", self._fake_fraction)
        summary.addRow("Уверенность (mean):", self._conf_mean)
        summary.addRow("Окна с речью:", self._speech_windows)
        layout.addLayout(summary)

        self._timeline = TimelineWidget()
        self._timeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._timeline, 1)

        layout.addWidget(QLabel("Сегменты предупреждений:"))
        self._segments = QListWidget()
        layout.addWidget(self._segments)

        note = QLabel(
            "Важно: это вероятностная оценка. Шум, пересжатие/VoIP и качество микрофона могут влиять на точность."
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        layout.addWidget(note)

    def _set_prob_style(self, color: Optional[str] = None) -> None:
        base = "font-size: 42px; font-weight: 800;"
        if color:
            self._prob_big.setStyleSheet(f"{base} color: {color};")
        else:
            self._prob_big.setStyleSheet(base)

    def _browse(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите аудиофайл",
            "",
            "Аудио (*.wav *.flac *.mp3 *.ogg *.m4a);;Все файлы (*)",
        )
        if path_str:
            self._path_edit.setText(path_str)

    def _start_analysis(self) -> None:
        if self._thread is not None:
            return

        path_str = self._path_edit.text().strip()
        if not path_str:
            self._browse()
            path_str = self._path_edit.text().strip()
        if not path_str:
            return

        path = Path(path_str)
        if not path.exists():
            QMessageBox.warning(self, "VoiceGuard", f"Файл не найден: {path}")
            return

        self._analysis = None
        self._export_btn.setEnabled(False)
        self._segments.clear()
        self._timeline.clear()
        self._chip_backend.setText("Режим: —")
        self._chip_duration.setText("Длительность: —")
        self._chip_speech.setText("Речь: —")
        self._chip_alerts.setText("Алерты: —")

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._analyze_btn.setEnabled(False)

        self._thread = QThread(self)
        self._worker = _FileAnalyzeWorker(path=path, config=self._config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._worker.error.connect(self._thread.quit)
        self._worker.error.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self._progress.setRange(0, 0)
            return
        self._progress.setRange(0, 100)
        self._progress.setValue(int(done * 100 / total))

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "VoiceGuard", message or "Неизвестная ошибка")

    def _on_finished(self, analysis_obj: object) -> None:
        analysis = analysis_obj if isinstance(analysis_obj, AnalysisResult) else None
        if analysis is None:
            QMessageBox.critical(self, "VoiceGuard", "Internal error: invalid analysis result.")
            return

        self._analysis = analysis
        backend = str(analysis.backend).lower()
        if backend == "onnx":
            backend_label = "ML модель (ONNX)"
        elif backend == "hf":
            backend_label = "ML модель (HuggingFace)"
        else:
            backend_label = "Демо (эвристика)"
        self._backend.setText(backend_label)
        self._chip_backend.setText(f"Режим: {backend_label}")
        backend_note = str(analysis.backend_note) if analysis.backend_note else ""
        if backend not in {"onnx", "hf"} and not backend_note:
            backend_note = "Для максимальной точности подключите ML‑модель (HF/ONNX)."
        self._backend_note.setText(backend_note)
        has_speech = int(analysis.summary.speech_windows) > 0
        if has_speech:
            self._p_mean.setText(format_percent(float(analysis.summary.p_fake_mean)))
            self._p_max.setText(format_percent(float(analysis.summary.p_fake_max)))
            self._fake_fraction.setText(format_percent(float(getattr(analysis.summary, "fake_fraction", 0.0))))
            self._conf_mean.setText(format_percent(float(analysis.summary.confidence_mean)))
        else:
            self._p_mean.setText("—")
            self._p_max.setText("—")
            self._fake_fraction.setText("—")
            self._conf_mean.setText("—")

        self._speech_windows.setText(f"{analysis.summary.speech_windows}/{analysis.summary.total_windows}")
        self._chip_duration.setText(f"Длительность: {analysis.summary.duration_sec:.1f}s")
        self._chip_speech.setText(
            f"Речь: {analysis.summary.speech_windows}/{analysis.summary.total_windows}"
        )
        self._chip_alerts.setText(f"Алерты: {len(analysis.summary.alert_segments)}")

        if not has_speech:
            self._prob_big.setText("—")
            self._set_prob_style()
            self._verdict_title.setText("Речь не обнаружена")
            self._verdict_subtitle.setText(
                "Файл слишком тихий или содержит в основном музыку/шум. Попробуйте другой фрагмент или увеличьте громкость."
            )
        else:
            p_overall = float(analysis.summary.p_fake_overall)
            verdict = make_verdict(
                p_overall,
                confidence=float(analysis.summary.confidence_mean),
                threshold=float(self._config.alert_threshold),
            )

            # If overall looks "real", but there are sustained high-risk segments,
            # show a cautionary verdict to avoid missing partial impersonation.
            if analysis.summary.alert_segments and p_overall < float(self._config.alert_threshold) * 0.60:
                verdict = Verdict(
                    title="В записи есть подозрительные фрагменты (возможен ИИ)",
                    subtitle="В целом запись может быть реальной, но отдельные сегменты выглядят как сгенерированные ИИ. Проверьте таймлайн.",
                    color="#f59e0b",  # amber-500
                )

            self._prob_big.setText(format_percent(p_overall))
            self._set_prob_style(verdict.color)
            self._verdict_title.setText(verdict.title)
            self._verdict_subtitle.setText(verdict.subtitle)

        times: list[float] = []
        values: list[Optional[float]] = []
        for w in analysis.windows:
            times.append(float(w.t_end))
            v = float(w.result.p_fake)
            values.append(None if (not w.result.is_speech or v != v) else v)

        segments = [TimeSegment(start_sec=s.start_sec, end_sec=s.end_sec) for s in analysis.summary.alert_segments]
        self._timeline.set_data(
            times=times,
            values=values,
            threshold=float(self._config.alert_threshold),
            segments=segments,
        )

        self._segments.clear()
        if analysis.summary.alert_segments:
            for seg in analysis.summary.alert_segments:
                self._segments.addItem(f"{seg.start_sec:.2f}s — {seg.end_sec:.2f}s")
        else:
            self._segments.addItem("—")

        self._export_btn.setEnabled(True)

    def _on_thread_finished(self) -> None:
        self._progress.setVisible(False)
        self._analyze_btn.setEnabled(True)
        self._thread = None
        self._worker = None

    def _export_report(self) -> None:
        if self._analysis is None:
            return

        out_dir = ensure_reports_dir(base_dir=None, reports_dir=str(self._config.storage.reports_dir))
        stem = default_report_stem()
        default_path = (out_dir / f"{stem}.html").as_posix()

        out_path, selected = QFileDialog.getSaveFileName(
            self,
            "Экспорт отчёта",
            default_path,
            "HTML (*.html);;JSON (*.json)",
        )
        if not out_path:
            return

        path = Path(out_path)
        try:
            if path.suffix.lower() == ".json" or selected.lower().startswith("json"):
                if path.suffix.lower() != ".json":
                    path = path.with_suffix(".json")
                write_json_report(self._analysis, path)
            else:
                if path.suffix.lower() != ".html":
                    path = path.with_suffix(".html")
                write_html_report(self._analysis, path)
        except Exception as exc:
            QMessageBox.critical(self, "VoiceGuard", str(exc))
            return

        QMessageBox.information(self, "VoiceGuard", f"Сохранено: {path}")

    def dragEnterEvent(self, event) -> None:  # noqa: ANN001
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: ANN001
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self._path_edit.setText(path)
