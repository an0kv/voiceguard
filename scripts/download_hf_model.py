from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace anti-spoofing / deepfake audio model snapshot for VoiceGuard."
    )
    parser.add_argument(
        "--repo-id",
        default="Shahzaib-Arshad/deepfake_audio_detection",
        help="HuggingFace model repo id (default: Shahzaib-Arshad/deepfake_audio_detection)",
    )
    parser.add_argument(
        "--out-dir",
        default="models/hf_shahzaib_deepfake",
        help="Destination directory inside the project (default: models/hf_shahzaib_deepfake)",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Optional git revision / tag / commit SHA.",
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "huggingface_hub is required. Install ML deps: pip install -r requirements-ml.txt"
        ) from exc

    allow = [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "*.txt",
        "*.md",
    ]

    snapshot_download(
        repo_id=str(args.repo_id),
        repo_type="model",
        revision=str(args.revision) if args.revision else None,
        local_dir=out_dir.as_posix(),
        local_dir_use_symlinks=False,
        allow_patterns=allow,
    )

    print(f"Downloaded to: {out_dir}")
    print("Next: set `model.backend: auto` (or `hf`) and `model.hf_local_dir` to this directory in config.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
