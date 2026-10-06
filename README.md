<h1 align="center">KOMOS</h1>

<p align="center"><strong>Automatic Mean Opinion Score Estimation for Korean Speech</strong></p>

<p align="center">
  🤗 <a href="https://huggingface.co/upskyy/komos"><strong>Hugging Face Model</strong></a>
</p>

KOMOS is a Korean mean opinion score predictor for overall speech quality across read and spontaneous speech. It combines a fully fine-tuned Whisper-medium encoder with a listener-conditioned prediction head and returns a score on the 1–5 MOS scale.

## Highlights

- Trained with 5,000 ratings from 29 native Korean listeners for 1,000 utterances.
- Covers real recordings, synthesized speech, and degraded speech.
- Supports PyTorch and ONNX Runtime inference.
- Uses a fixed 10-second input protocol for utterance-level scoring.

## Results

### Overall test performance

Utterance-level correlation with human MOS on the 100-utterance test set. UTMOS v1 and v2 are evaluated as English-trained zero-shot baselines, while KOMOS is fine-tuned on Korean quality ratings. Higher is better.

| Model | Test SRCC | Test LCC |
| --- | ---: | ---: |
| UTMOS v1 | 0.703 | 0.798 |
| UTMOS v2 | 0.767 | 0.858 |
| **KOMOS** | **0.900** | **0.950** |

KOMOS improves test SRCC by 0.133 over the stronger UTMOS v2 baseline, showing a stronger monotonic and linear association with human judgments.

### Speaker- and text-disjoint evaluation

Utterance-level SRCC on test subsets whose speakers or texts are absent from both the training and validation sets.

| Model | Overall (n=100) | Speaker-disjoint (n=34) | Text-disjoint (n=55) |
| --- | ---: | ---: | ---: |
| UTMOS v1 | 0.703 | 0.818 | 0.719 |
| UTMOS v2 | 0.767 | 0.823 | 0.778 |
| **KOMOS** | **0.900** | **0.951** | **0.878** |

KOMOS maintains the highest rank correlation on both subsets, supporting generalization to held-out speakers and texts within the main seen-system setting.

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/upskyy/komos.git
cd komos
uv sync --no-dev
```

## Quickstart

### PyTorch

```python
import torch

from komos import KOMOS

model = KOMOS.from_pretrained("upskyy/komos", dtype=torch.float32).eval()
score = model.score("audio.wav")
print(score)
```

Move the model to CUDA with `model.to("cuda")` when GPU inference is needed.

### ONNX Runtime

```bash
uv sync --no-dev --extra onnx
```

```python
from komos.onnx import KOMOSOnnx

model = KOMOSOnnx.from_pretrained("upskyy/komos")
score = model.score("audio.wav")
print(score)
```

### torch.hub

```python
import torch

model = torch.hub.load("upskyy/komos:main", "komos", trust_repo=True)
print(model.score("audio.wav"))
```

## Inference

`score()` accepts an audio path or a mono NumPy array or PyTorch tensor with an explicit sample rate. File inputs are downmixed to mono and resampled to 16 kHz.

- `score()` evaluates up to the first 10 seconds and returns a validation-calibrated score clipped to the 1–5 range.
- `score_long_form()` accepts up to 60 seconds and aggregates non-overlapping 10-second chunks by duration.
- Inputs longer than 60 seconds should be processed in external batches.

For comparable system-level results, score each utterance independently with `score()` and report the unweighted mean.

## Limitations

KOMOS predicts overall speech quality. It does not independently verify pronunciation accuracy or intelligibility. The main test split is a seen-system setting, and long-form scoring has not been validated against human MOS for long recordings.

This repository provides inference code and model weights. The training dataset and experiment artifacts are not included.

## License

KOMOS is released under the MIT License. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
