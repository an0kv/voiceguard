from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)


class OnnxModel:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(str(self._path))

        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is required for model.backend=onnx; install `onnxruntime`."
            ) from exc

        # Keep it CPU-only by default for portability.
        sess_opts = ort.SessionOptions()
        self._session = ort.InferenceSession(
            self._path.as_posix(), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )

        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs.")
        self._input_name = str(inputs[0].name)

        outputs = self._session.get_outputs()
        self._output_name: Optional[str] = str(outputs[0].name) if outputs else None

    @property
    def path(self) -> Path:
        return self._path

    def predict(self, log_mel: np.ndarray) -> float:
        x = log_mel.astype(np.float32, copy=False)
        if x.ndim == 2:
            x = x[None, :, :]
        if x.ndim != 3:
            raise ValueError(f"Expected log_mel with shape (n_mels, n_frames) or (1, n_mels, n_frames); got {x.shape}")

        outputs = self._session.run(
            None if self._output_name is None else [self._output_name],
            {self._input_name: x},
        )
        if not outputs:
            raise RuntimeError("ONNX inference returned no outputs.")

        y = np.array(outputs[0])
        y = y.astype(np.float32, copy=False)

        # Accept a few common heads:
        # - probability scalar (sigmoid): (1,) or (1,1)
        # - logits/probs for 2 classes: (1,2) -> take index 1 as 'fake'
        if y.size == 1:
            p = float(y.reshape(-1)[0])
            return float(np.clip(p, 0.0, 1.0))

        if y.ndim >= 1 and y.shape[-1] == 2:
            probs = _softmax(y, axis=-1)
            p = float(probs.reshape(-1, 2)[0, 1])
            return float(np.clip(p, 0.0, 1.0))

        # Fallback: take first element and clamp.
        p = float(y.reshape(-1)[0])
        return float(np.clip(p, 0.0, 1.0))

