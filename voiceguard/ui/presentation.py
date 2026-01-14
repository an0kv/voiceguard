from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Verdict:
    title: str
    subtitle: str
    color: str  # hex


def format_percent(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return f"{max(0.0, min(1.0, float(p))) * 100.0:.0f}%"


def confidence_label(confidence: float) -> str:
    c = max(0.0, min(1.0, float(confidence)))
    if c < 0.34:
        return "низкая"
    if c < 0.67:
        return "средняя"
    return "высокая"


def make_verdict(p_fake: Optional[float], *, confidence: float, threshold: float) -> Verdict:
    if p_fake is None:
        return Verdict(
            title="Нет данных",
            subtitle="Говорите в микрофон или выберите файл для анализа.",
            color="#94a3b8",  # slate-400
        )

    p = max(0.0, min(1.0, float(p_fake)))
    th = max(0.0, min(1.0, float(threshold)))

    conf_txt = confidence_label(confidence)
    conf_pct = format_percent(confidence)

    if p >= th:
        return Verdict(
            title="Высокая вероятность: голос сгенерирован ИИ (не человек)",
            subtitle=f"Уверенность: {conf_txt} ({conf_pct}). Это вероятностная оценка, не 100% доказательство.",
            color="#ef4444",  # red-500
        )

    if p >= th * 0.60:
        return Verdict(
            title="Есть признаки синтетического голоса (возможен ИИ)",
            subtitle=f"Уверенность: {conf_txt} ({conf_pct}). Проверьте источник/условия записи.",
            color="#f59e0b",  # amber-500
        )

    return Verdict(
        title="Похоже на реальный голос человека (не ИИ)",
        subtitle=f"Уверенность: {conf_txt} ({conf_pct}). Всё равно учитывайте контекст и риски мошенничества.",
        color="#22c55e",  # green-500
    )
