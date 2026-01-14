from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _find_class_index(id2label: dict[int, str], *, want: str) -> Optional[int]:
    want_l = want.lower()
    for idx, label in id2label.items():
        if want_l in str(label).lower():
            return int(idx)
    return None


def _pick_fake_index(id2label: dict[int, str]) -> int:
    # Common class names across deepfake / anti-spoofing models.
    fake_keywords = ("fake", "spoof", "synthetic", "deepfake", "ai", "clone", "tts")
    real_keywords = ("real", "bonafide", "bona-fide", "human", "genuine")

    for idx, label in id2label.items():
        l = str(label).lower()
        if any(k in l for k in fake_keywords):
            return int(idx)

    real_idx = None
    for idx, label in id2label.items():
        l = str(label).lower()
        if any(k in l for k in real_keywords):
            real_idx = int(idx)
            break

    # If it's a binary head and we found "real", the other class is fake/spoof.
    if real_idx is not None and len(id2label) == 2:
        other = [i for i in id2label.keys() if int(i) != int(real_idx)]
        if other:
            return int(other[0])

    # Last resort: assume index 0 is "fake/spoof".
    return 0


@dataclass(frozen=True)
class HfPrediction:
    p_fake: float
    model_confidence: float


class HfAudioClassifier:
    def __init__(
        self,
        *,
        repo_id: str,
        local_dir: Optional[Path] = None,
        revision: str = "",
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "HF backend requires `torch` + `transformers`. Install `requirements-ml.txt`."
            ) from exc

        self._torch = torch
        self._repo_id = str(repo_id)
        self._local_dir = Path(local_dir) if local_dir is not None else None
        self._revision = str(revision or "")

        model_source = self._local_dir.as_posix() if self._local_dir is not None else self._repo_id
        kwargs = {"revision": self._revision} if self._revision else {}

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_source, **kwargs)
        self._model = AutoModelForAudioClassification.from_pretrained(model_source, **kwargs)
        self._model.eval()
        self._model.to("cpu")

        # Normalize id2label to int->str
        id2label_raw = getattr(self._model.config, "id2label", {}) or {}
        id2label: dict[int, str] = {}
        for k, v in id2label_raw.items():
            try:
                id2label[int(k)] = str(v)
            except Exception:
                continue
        if not id2label:
            # Default LABEL_0..N-1
            num_labels = int(getattr(self._model.config, "num_labels", 2) or 2)
            id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        self._id2label = id2label
        self._fake_idx = _pick_fake_index(self._id2label)

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @property
    def local_dir(self) -> Optional[Path]:
        return self._local_dir

    def predict(self, audio_16k: np.ndarray, *, sample_rate: int) -> HfPrediction:
        x = audio_16k.astype(np.float32, copy=False).reshape(-1)
        if x.size == 0:
            return HfPrediction(p_fake=0.0, model_confidence=0.0)

        inputs = self._feature_extractor(x, sampling_rate=int(sample_rate), return_tensors="pt")
        with self._torch.no_grad():
            out = self._model(**inputs)
        logits = out.logits
        probs = self._torch.softmax(logits, dim=-1)[0]

        p_fake = float(probs[int(self._fake_idx)].item())
        conf = float(self._torch.max(probs).item())

        return HfPrediction(
            p_fake=float(np.clip(p_fake, 0.0, 1.0)),
            model_confidence=float(np.clip(conf, 0.0, 1.0)),
        )
