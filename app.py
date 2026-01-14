from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from voiceguard.config import load_config
from voiceguard.ui.main_window import MainWindow
from voiceguard.ui.theme import apply_theme, load_theme_preference


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("VoiceGuard")
    app.setApplicationName("VoiceGuard Desktop")
    apply_theme(app, load_theme_preference())

    config_path = Path(__file__).with_name("config.yaml")
    config = load_config(config_path)

    window = MainWindow(config=config)
    window.resize(1100, 700)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
