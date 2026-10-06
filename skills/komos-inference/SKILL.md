---
name: komos-inference
description: Install, run, or integrate KOMOS for Korean speech quality MOS prediction. Use when scoring audio with KOMOS, adding KOMOS inference to Python code, or explaining KOMOS outputs and audio handling.
---

# KOMOS Inference

Use the repository's public inference API and treat `README.md` as the source of truth for installation and current examples. Inspect `src/komos/` only when implementation details are needed.

## Run inference

Prefer the Hugging Face loading path unless the user specifically requests `torch.hub`:

### PyTorch

```python
from komos import KOMOS

model = KOMOS.from_pretrained("upskyy/komos").eval()
score = model.score("audio.wav")
print(score)
```

Move the model to CUDA with `model.to("cuda")` when GPU inference is needed. Do not enable remote-code execution; the inference implementation is provided by this package.

For waveform input, pass a mono NumPy array or PyTorch tensor and provide `sr`. File input is automatically downmixed to mono and resampled to 16 kHz.

### ONNX Runtime

Install the `onnx` package extra:

```bash
uv sync --no-dev --extra onnx
```

Use the supported wrapper rather than invoking the graph directly:

```python
from komos.onnx import KOMOSOnnx

model = KOMOSOnnx.from_pretrained("upskyy/komos")
score = model.score("audio.wav")
print(score)
```

The opset 17 graph has a dynamic batch dimension and fixed 10-second feature and attention-mask dimensions. Its `raw_score` output is uncalibrated, while `KOMOSOnnx.score()` applies the model metadata's calibration and clipping.

### torch.hub

```python
import torch

model = torch.hub.load("upskyy/komos:main", "komos", trust_repo=True)
print(model.score("audio.wav"))
```

## Duration behavior

- KOMOS uses a 10-second input window rather than Whisper's commonly used 30-second window.
- Use `score()` for the default utterance-level KOMOS score. It scores the first 10 seconds and matches the training and evaluation input processing.
- Use `score_long_form()` for recordings longer than 10 seconds and up to 60 seconds. Longer inputs raise an error and should be processed in external batches.
- Long-form mode splits audio into non-overlapping 10-second chunks and computes a duration-weighted mean of raw chunk scores before calibration.
- State that long-form scoring is a convenience mode that has not been validated against human MOS for long recordings when this affects interpretation.

Unless otherwise specified, “KOMOS score” refers to the default utterance-level inference protocol, which uses up to the first 10 seconds of each utterance.

For comparable system-level reporting, score each utterance independently using the default inference mode and report the unweighted mean across utterances. If long-form scoring is used, please report it explicitly, as it follows a different inference protocol from the default utterance-level KOMOS score.

## Interpret scores

- `score()` returns a validation-calibrated prediction clipped to the 1--5 MOS range. Do not apply calibration or clipping again.
- Higher scores indicate greater predicted overall speech quality. Do not present them as verified pronunciation accuracy or intelligibility scores.
