from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import soundfile as sf  # pyright: ignore[reportMissingTypeStubs]
import torch
from numpy.typing import NDArray
from torchaudio.functional import resample  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import assert_never

AudioInput: TypeAlias = str | Path | NDArray[np.float32] | NDArray[np.float64] | torch.Tensor
SAMPLE_RATE: Final = 16000
MIN_SAMPLES: Final = 2


class AudioInputError(ValueError):
    """Invalid waveform or sample rate."""


class AudioTruncationWarning(UserWarning):
    """Only the initial segment of an input is scored."""


def load_audio(audio: AudioInput, sr: int | None) -> NDArray[np.float32]:
    """Load mono audio and resample to 16 kHz."""
    match audio:
        case str() | Path():
            samples, input_rate = sf.read(audio, dtype="float32", always_2d=True)
            waveform = torch.as_tensor(samples.mean(axis=1))
        case torch.Tensor():
            waveform = audio.detach().to(device="cpu", dtype=torch.float32)
            input_rate = sr
        case np.ndarray():
            waveform = torch.as_tensor(np.array(audio, dtype=np.float32, copy=True))
            input_rate = sr
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(unreachable)
    if input_rate is None or input_rate <= 0:
        message = "Provide a positive sr for waveform input."
        raise AudioInputError(message)
    if waveform.ndim != 1 or waveform.numel() < MIN_SAMPLES:
        message = "Expected a mono waveform with at least two samples."
        raise AudioInputError(message)
    if not bool(torch.isfinite(waveform).all()):
        message = "Audio must contain only finite samples."
        raise AudioInputError(message)
    if input_rate != SAMPLE_RATE:
        waveform = resample(waveform, input_rate, SAMPLE_RATE)
    if waveform.numel() < MIN_SAMPLES:
        message = "Audio is too short after resampling."
        raise AudioInputError(message)
    return np.asarray(waveform, dtype=np.float32)
