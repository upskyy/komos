# KOMOS

Inference code for **KOMOS: Automatic Mean Opinion Score Estimation for Korean Speech**.

KOMOS predicts Korean speech naturalness on a 1–5 MOS scale using a Whisper-medium encoder and a listener-conditioned prediction head. It is fine-tuned on 1,000 utterances with 5,000 human ratings covering read and spontaneous speech.

The paper reports test SRCC/LCC of **0.900/0.950**, averaged over three training seeds. The released checkpoint is the **seed 42** model, not an ensemble.

Model: [upskyy/komos](https://huggingface.co/upskyy/komos)

## Install

Python 3.11 or later is required. The package is not published on PyPI yet.

```bash
git clone https://github.com/upskyy/komos.git
cd komos
uv sync --no-dev
```

While the repositories are private, authenticate GitHub before cloning and run `uv run hf auth login` with a Hugging Face account that has model access.

## Hugging Face

```python
from komos import KOMOS

model = KOMOS.from_pretrained("upskyy/komos").eval()
score = model.score("audio.wav")
print(score)
```

## torch.hub

Install the package first using the steps above.

```python
import torch

model = torch.hub.load("upskyy/komos:main", "komos", trust_repo=True)
print(model.score("audio.wav"))
```

For a private GitHub repository, use the authenticated local clone:

```python
model = torch.hub.load(".", "komos", source="local")
```

## Audio input

`score()` accepts an audio path or a mono NumPy array/PyTorch tensor with an explicit sample rate:

```python
import soundfile as sf

audio, sr = sf.read("audio.wav", dtype="float32", always_2d=True)
print(model.score(audio.mean(axis=1), sr=sr))
```

File inputs are downmixed to mono and resampled to 16 kHz. By default, KOMOS scores only the first 10 seconds, matching the input processing used for training and evaluation. Longer inputs trigger a warning. The returned score describes the initial segment, not the entire recording.

For optional long-form scoring:

```python
score = model.score("long_audio.wav", long_form=True)
```

This mode splits audio into non-overlapping 10-second chunks and weights raw scores by each chunk's duration, including the final partial chunk. Validation-fitted affine calibration and clipping to [1, 5] are applied once after aggregation. This convenience mode has not been validated against human MOS for long recordings.

Both modes return calibrated scores. Use `model.to("cuda")` for GPU inference. The paper's correlation results use raw predictions, and the reported three-seed means are not reproduced by this single checkpoint alone.

Higher scores indicate greater predicted naturalness, not verified pronunciation accuracy. The main test split contains seen TTS systems. Unseen-system results in the paper are an author-screened diagnostic, not independent human-MOS validation.

Training code, datasets, and experiment artifacts are not included.

## Package layout

```text
src/komos/
  __init__.py
  komos.py     # KOMOS model and scoring
  config.py    # Model configuration
  audio.py     # Audio loading and resampling
```

The Hugging Face repository stores weights and configuration. Inference code is provided by this package, without remote-code execution.

Build the Python distribution with `uv build`. This does not publish it to PyPI.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
