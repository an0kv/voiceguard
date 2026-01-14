from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from voiceguard.analysis import AnalysisResult


def _dt_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def analysis_to_dict(analysis: AnalysisResult) -> dict[str, Any]:
    d = asdict(analysis)
    # Make floats JSON-friendly (NaN -> None).
    for w in d.get("windows", []):
        result = w.get("result", {})
        for k in ("p_fake", "p_fake_smooth"):
            v = result.get(k)
            if isinstance(v, float) and (v != v):  # NaN
                result[k] = None
    return d


def ensure_reports_dir(*, base_dir: Optional[Path], reports_dir: str) -> Path:
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    out = (root / reports_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def default_report_stem(*, prefix: str = "voiceguard") -> str:
    return f"{prefix}_{_dt_slug()}"


def write_json_report(analysis: AnalysisResult, path: Path) -> None:
    path = Path(path)
    payload = analysis_to_dict(analysis)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_html_report(analysis: AnalysisResult, path: Path) -> None:
    path = Path(path)
    summary = analysis.summary

    seg_html = ""
    if summary.alert_segments:
        seg_html = "<ul>"
        for seg in summary.alert_segments:
            seg_html += f"<li>{seg.start_sec:.2f}s — {seg.end_sec:.2f}s</li>"
        seg_html += "</ul>"
    else:
        seg_html = "<p>Нет.</p>"

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VoiceGuard Report</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    .k {{ color: #555; }}
    .card {{ border: 1px solid #eee; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    code, pre {{ background: #f7f7f7; border-radius: 8px; padding: 2px 6px; }}
    pre {{ padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>VoiceGuard Report</h1>
    <div class="card">
    <div><span class="k">Источник:</span> {analysis.source_kind} — <code>{analysis.source}</code></div>
    <div><span class="k">Время (UTC):</span> <code>{analysis.created_at}</code></div>
    <div><span class="k">Backend:</span> <code>{analysis.backend}</code></div>
    <div><span class="k">Backend note:</span> <code>{analysis.backend_note or "—"}</code></div>
    <div><span class="k">Длительность:</span> {summary.duration_sec:.2f}s</div>
    <div><span class="k">Окна:</span> {summary.total_windows} (speech: {summary.speech_windows})</div>
    <div><span class="k">Уверенность (mean):</span> {summary.confidence_mean:.3f}</div>
    <div><span class="k">Уверенность (min):</span> {summary.confidence_min:.3f}</div>
    <div><span class="k">p_fake overall:</span> {summary.p_fake_overall:.3f}</div>
    <div><span class="k">p_fake p95:</span> {summary.p_fake_p95:.3f}</div>
    <div><span class="k">p_fake mean:</span> {summary.p_fake_mean:.3f}</div>
    <div><span class="k">p_fake max:</span> {summary.p_fake_max:.3f}</div>
    <div><span class="k">Fake fraction (>= threshold):</span> {getattr(summary, "fake_fraction", 0.0):.3f}</div>
  </div>
  <div class="card">
    <h2>Сегменты алертов</h2>
    {seg_html}
  </div>
  <div class="card">
    <h2>Raw JSON</h2>
    <pre>{json.dumps(analysis_to_dict(analysis), ensure_ascii=False, indent=2)}</pre>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
