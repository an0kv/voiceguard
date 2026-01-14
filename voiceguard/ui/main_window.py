from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget

from voiceguard.config import AppConfig
from voiceguard.ui.file_tab import FileTab
from voiceguard.ui.live_tab import LiveTab
from voiceguard.ui.theme import THEME_DARK, THEME_LIGHT, apply_theme, load_theme_preference, save_theme_preference
from voiceguard.ui.widgets.timeline import TimelineWidget


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("VoiceGuard — защита от имитации голоса")
        self._theme_mode = load_theme_preference()

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setTabsClosable(False)
        tabs.setMovable(False)
        tabs.setElideMode(Qt.TextElideMode.ElideRight)

        tabs.addTab(FileTab(config=config), "Файл")
        tabs.addTab(LiveTab(config=config), "Онлайн")

        self.setCentralWidget(tabs)

        self.statusBar().showMessage("Локальная обработка. Результат — вероятностная оценка «ИИ vs человек».")

        view_menu = self.menuBar().addMenu("Вид")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        light_action = QAction("Светлая тема", self, checkable=True)
        dark_action = QAction("Тёмная тема", self, checkable=True)
        theme_group.addAction(light_action)
        theme_group.addAction(dark_action)
        view_menu.addAction(light_action)
        view_menu.addAction(dark_action)
        if self._theme_mode == THEME_DARK:
            dark_action.setChecked(True)
        else:
            light_action.setChecked(True)
        light_action.triggered.connect(lambda: self._set_theme(THEME_LIGHT))
        dark_action.triggered.connect(lambda: self._set_theme(THEME_DARK))

        help_menu = self.menuBar().addMenu("Справка")
        about = help_menu.addAction("О VoiceGuard")
        about.triggered.connect(self._show_about)

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "О VoiceGuard",
            "VoiceGuard оценивает вероятность того, что голос сгенерирован ИИ (TTS/voice clone).\n\n"
            "Важно: это НЕ 100% доказательство. На результат влияют шум, кодеки/VoIP, микрофон, пересжатие.\n"
            "По умолчанию приложение не сохраняет и не отправляет аудио в облако.",
        )

    def _set_theme(self, mode: str) -> None:
        if mode == self._theme_mode:
            return
        self._theme_mode = mode
        save_theme_preference(mode)
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, mode)
        for timeline in self.findChildren(TimelineWidget):
            timeline.update()
