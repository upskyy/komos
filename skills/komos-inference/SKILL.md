---
name: komos-inference
description: Install, run, or integrate KOMOS for Korean speech quality MOS prediction. Use when scoring audio with KOMOS, adding KOMOS inference to Python code, or explaining KOMOS outputs and audio handling.
---

# KOMOS Inference

Use the repository's public inference API and treat `README.md` as the source of truth for installation and current examples. Inspect `src/komos/` only when implementation details are needed.

## Install

Python 3.10 or later is required. For ordinary use, install from PyPI:

```bash
pip install komos
```

For ONNX Runtime inference, install the optional extra:

```bash
pip install "komos[onnx]"
```

When working from a repository checkout, use `uv sync --no-dev` for PyTorch or `uv sync --no-dev --extra onnx` for ONNX Runtime.

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

### torch.hub

```python
import torch

model = torch.hub.load("upskyy/komos:v0.0.2", "komos", trust_repo=True)
print(model.score("audio.wav"))
```

### ONNX Runtime

Use the supported wrapper rather than invoking the graph directly:

```python
from komos.onnx import KOMOSOnnx

model = KOMOSOnnx.from_pretrained("upskyy/komos")
score = model.score("audio.wav")
print(score)
```

The opset 17 graph has a dynamic batch dimension and fixed 10-second feature and attention-mask dimensions. Its `raw_score` output is uncalibrated, while `KOMOSOnnx.score()` applies the model metadata's calibration and clipping.

## Duration behavior

- KOMOS uses a 10-second input window rather than Whisper's commonly used 30-second window.
- The paper reports results using the utterance-level `score()` protocol, which evaluates up to the first 10 seconds.
- `score_long_form()` is the supported interface for recordings longer than 10 seconds and up to 60 seconds. Longer inputs raise an error and should be processed in external batches.
- Long-form mode splits audio into non-overlapping 10-second chunks and computes a duration-weighted mean of raw chunk scores before calibration.

## Interpret scores

- `score()` returns a validation-calibrated prediction clipped to the 1--5 MOS range. Do not apply calibration or clipping again.
- Higher scores indicate greater predicted overall speech quality. Do not present them as verified pronunciation accuracy or intelligibility scores.
