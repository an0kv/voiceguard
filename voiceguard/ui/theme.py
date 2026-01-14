from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

THEME_LIGHT = "light"
THEME_DARK = "dark"
_THEME_KEY = "ui/theme"


@dataclass(frozen=True)
class ThemeSpec:
    name: str
    window: str
    text: str
    base: str
    alt_base: str
    tooltip_base: str
    tooltip_text: str
    button: str
    button_text: str
    bright_text: str
    accent: str
    accent_hover: str
    accent_pressed: str
    border: str
    muted: str
    hero: str
    chip_bg: str
    chip_text: str
    tab_bg: str
    tab_text: str
    tab_selected_bg: str
    tab_selected_text: str
    group_bg: str
    group_title: str
    card_bg: str
    stat_bg: str
    disabled_bg: str
    disabled_text: str


LIGHT_THEME = ThemeSpec(
    name=THEME_LIGHT,
    window="#f6f7fb",
    text="#0f172a",
    base="#ffffff",
    alt_base="#f1f5f9",
    tooltip_base="#0f172a",
    tooltip_text="#f8fafc",
    button="#ffffff",
    button_text="#0f172a",
    bright_text="#ef4444",
    accent="#0ea5a4",
    accent_hover="#0f8f8a",
    accent_pressed="#0f766e",
    border="#e2e8f0",
    muted="#64748b",
    hero="#334155",
    chip_bg="#f1f5f9",
    chip_text="#475569",
    tab_bg="#f1f5f9",
    tab_text="#475569",
    tab_selected_bg="#ffffff",
    tab_selected_text="#0f172a",
    group_bg="#ffffff",
    group_title="#64748b",
    card_bg="#ffffff",
    stat_bg="#f8fafc",
    disabled_bg="#f1f5f9",
    disabled_text="#94a3b8",
)

DARK_THEME = ThemeSpec(
    name=THEME_DARK,
    window="#0f1116",
    text="#f1f5f9",
    base="#151a23",
    alt_base="#1b2230",
    tooltip_base="#151a23",
    tooltip_text="#f1f5f9",
    button="#1b2230",
    button_text="#f1f5f9",
    bright_text="#f87171",
    accent="#22c1a7",
    accent_hover="#1fb19a",
    accent_pressed="#159a87",
    border="#2b3445",
    muted="#94a3b8",
    hero="#e2e8f0",
    chip_bg="#151a23",
    chip_text="#cbd5e1",
    tab_bg="#151a23",
    tab_text="#a8b0bf",
    tab_selected_bg="#1b2230",
    tab_selected_text="#f1f5f9",
    group_bg="#161b22",
    group_title="#a8b0bf",
    card_bg="#161b22",
    stat_bg="#151a23",
    disabled_bg="#1b2230",
    disabled_text="#94a3b8",
)


def load_theme_preference() -> str:
    settings = QSettings()
    value = str(settings.value(_THEME_KEY, THEME_LIGHT))
    return value if value in {THEME_LIGHT, THEME_DARK} else THEME_LIGHT


def save_theme_preference(mode: str) -> None:
    settings = QSettings()
    settings.setValue(_THEME_KEY, mode)


def apply_theme(app: QApplication, mode: str = THEME_LIGHT) -> None:
    spec = DARK_THEME if mode == THEME_DARK else LIGHT_THEME
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(spec.window))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(spec.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(spec.base))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(spec.alt_base))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(spec.tooltip_base))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(spec.tooltip_text))
    palette.setColor(QPalette.ColorRole.Text, QColor(spec.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(spec.button))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(spec.button_text))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(spec.bright_text))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(spec.accent))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    app.setStyleSheet(_build_stylesheet(spec))


def _build_stylesheet(spec: ThemeSpec) -> str:
    return f"""
        * {{ font-size: 12.5px; font-family: "Avenir Next", "SF Pro Text", "Segoe UI Variable", "Noto Sans"; }}
        QMainWindow {{
          background: {spec.window};
        }}
        QLabel {{ color: {spec.text}; }}
        QLabel[muted="true"] {{ color: {spec.muted}; }}
        QLabel[tone="hero"] {{ color: {spec.hero}; }}

        QLineEdit, QComboBox, QDoubleSpinBox {{
          background: {spec.base};
          border: 1px solid {spec.border};
          border-radius: 8px;
          padding: 6px 10px;
        }}
        QComboBox::drop-down {{ border: 0px; }}

        QPushButton {{
          background: {spec.accent};
          border: 1px solid {spec.accent_pressed};
          color: #ffffff;
          border-radius: 8px;
          padding: 7px 12px;
        }}
        QPushButton:hover {{ background: {spec.accent_hover}; }}
        QPushButton:pressed {{ background: {spec.accent_pressed}; }}
        QPushButton:disabled {{
          background: {spec.disabled_bg};
          border: 1px solid {spec.border};
          color: {spec.disabled_text};
        }}
        QPushButton[secondary="true"] {{
          background: transparent;
          border: 1px solid {spec.border};
          color: {spec.text};
        }}
        QPushButton[secondary="true"]:hover {{
          background: {spec.alt_base};
        }}

        QProgressBar {{
          border: 1px solid {spec.border};
          border-radius: 8px;
          background: {spec.base};
          text-align: center;
          padding: 2px;
        }}
        QProgressBar::chunk {{
          background: {spec.accent};
          border-radius: 6px;
        }}

        QListWidget {{
          background: {spec.base};
          border: 1px solid {spec.border};
          border-radius: 10px;
          padding: 6px;
        }}

        QTabWidget::pane {{
          border: 1px solid {spec.border};
          border-radius: 10px;
          top: -1px;
          background: {spec.base};
        }}
        QTabBar::tab {{
          background: {spec.tab_bg};
          border: 1px solid {spec.border};
          border-bottom: none;
          padding: 9px 14px;
          border-top-left-radius: 8px;
          border-top-right-radius: 8px;
          margin-right: 4px;
          color: {spec.tab_text};
        }}
        QTabBar::tab:selected {{
          background: {spec.tab_selected_bg};
          color: {spec.tab_selected_text};
        }}

        QGroupBox {{
          border: 1px solid {spec.border};
          border-radius: 12px;
          margin-top: 12px;
          padding: 10px;
          background: {spec.group_bg};
        }}
        QGroupBox::title {{
          subcontrol-origin: margin;
          subcontrol-position: top left;
          padding: 0 6px;
          color: {spec.group_title};
        }}

        QFrame[card="true"] {{
          background: {spec.card_bg};
          border: 1px solid {spec.border};
          border-radius: 14px;
        }}
        QFrame[stat="true"] {{
          background: {spec.stat_bg};
          border: 1px solid {spec.border};
          border-radius: 12px;
        }}
        QLabel[chip="true"] {{
          background: {spec.chip_bg};
          border: 1px solid {spec.border};
          border-radius: 999px;
          padding: 3px 8px;
          color: {spec.chip_text};
        }}
        QCheckBox {{ spacing: 8px; }}
        QCheckBox::indicator {{
          width: 16px;
          height: 16px;
          border-radius: 4px;
          border: 1px solid {spec.border};
          background: {spec.base};
        }}
        QCheckBox::indicator:checked {{
          background: {spec.accent};
          border: 1px solid {spec.accent_pressed};
        }}
        """
