# VoiceGuard Android (prototype)

This folder contains a minimal Android app that performs live audio capture and runs the same heuristic detector as the desktop MVP.

## Features

- Live detection from **microphone** (voice communication source).
- Live detection from **system audio** via **AudioPlaybackCapture** (Android 10+).
- Voice‑only capture toggle for Zoom/Discord (voice communication usages only).
- Optional audio effects: noise suppressor, echo canceller, auto gain.
- Preset modes (Call/Meeting/Studio) with tuned thresholds.
- Sliding-window analysis with smoothing and alert hold.

## Build & run

1) Open `android/` in Android Studio.
2) Let Gradle sync.
3) Run on a device with Android 10+ for system-audio capture.

## Notes & limitations

- AudioPlaybackCapture works only if the target app allows capture; many VoIP apps may block it.
- Capturing phone calls is restricted by Android policy; VoIP apps may or may not be capturable.
- Detection uses a **heuristic** (no ML model). For higher accuracy, integrate a TFLite/ONNX model in `VoiceGuardEngine`.

## Permissions

- `RECORD_AUDIO` is required for both microphone and playback capture.
- Playback capture requires user consent via the system screen-capture prompt.
