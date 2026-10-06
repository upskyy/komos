from dataclasses import dataclass
from math import ceil, gcd
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from scipy.signal import resample_poly
from typing_extensions import assert_never

AudioInput: TypeAlias = str | Path | NDArray[np.float32] | NDArray[np.float64] | torch.Tensor
SAMPLE_RATE: Final = 16000
MIN_SAMPLES: Final = 2
MAX_LONG_FORM_SECONDS: Final = 60.0


class AudioInputError(ValueError):
    """Invalid waveform or sample rate."""


class AudioTruncationWarning(UserWarning):
    """Only the initial segment of an input is scored."""


@dataclass(frozen=True, slots=True)
class AudioLimit:
    seconds: float
    truncate: bool


@dataclass(frozen=True, slots=True)
class LoadedAudio:
    waveform: NDArray[np.float32]
    truncated: bool


def _limited_frames(
    total_frames: int, input_rate: int, limit: AudioLimit | None
) -> tuple[int | None, bool]:
    if limit is None:
        return None, False
    max_frames = ceil(limit.seconds * input_rate)
    if total_frames <= max_frames:
        return None, False
    if limit.truncate:
        return max_frames, True
    message = (
        f"Audio exceeds the {limit.seconds:g}-second score_long_form() limit. "
        "Process longer recordings in external batches."
    )
    raise AudioInputError(message)


def _require_input_rate(input_rate: int | None) -> int:
    if input_rate is None or input_rate <= 0:
        message = "Provide a positive sr for waveform input."
        raise AudioInputError(message)
    return input_rate


def _load_file(audio: str | Path, limit: AudioLimit | None) -> tuple[torch.Tensor, int, bool]:
    audio_info = sf.info(audio)
    input_rate = audio_info.samplerate
    frame_limit, truncated = _limited_frames(audio_info.frames, input_rate, limit)
    samples, _ = sf.read(
        audio,
        frames=-1 if frame_limit is None else frame_limit,
        dtype="float32",
        always_2d=True,
    )
    return torch.as_tensor(samples.mean(axis=1)), input_rate, truncated


def _load_waveform(
    audio: NDArray[np.float32] | NDArray[np.float64] | torch.Tensor,
    sr: int | None,
    limit: AudioLimit | None,
) -> tuple[torch.Tensor, int, bool]:
    input_rate = _require_input_rate(sr)
    match audio:
        case torch.Tensor():
            if audio.ndim != 1 or audio.numel() < MIN_SAMPLES:
                message = "Expected a mono waveform with at least two samples."
                raise AudioInputError(message)
            frame_limit, truncated = _limited_frames(audio.numel(), input_rate, limit)
            waveform = audio[:frame_limit].detach().to(device="cpu", dtype=torch.float32)
        case np.ndarray():
            if audio.ndim != 1 or audio.size < MIN_SAMPLES:
                message = "Expected a mono waveform with at least two samples."
                raise AudioInputError(message)
            frame_limit, truncated = _limited_frames(audio.size, input_rate, limit)
            waveform = torch.as_tensor(np.array(audio[:frame_limit], dtype=np.float32, copy=True))
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(unreachable)
    return waveform, input_rate, truncated


def load_audio(audio: AudioInput, sr: int | None, limit: AudioLimit | None = None) -> LoadedAudio:
    """Load mono audio and resample to 16 kHz."""
    match audio:
        case str() | Path():
            waveform, input_rate, truncated = _load_file(audio, limit)
        case torch.Tensor() | np.ndarray():
            waveform, input_rate, truncated = _load_waveform(audio, sr, limit)
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(unreachable)
    if waveform.ndim != 1 or waveform.numel() < MIN_SAMPLES:
        message = "Expected a mono waveform with at least two samples."
        raise AudioInputError(message)
    if not bool(torch.isfinite(waveform).all()):
        message = "Audio must contain only finite samples."
        raise AudioInputError(message)
    if input_rate != SAMPLE_RATE:
        common_divisor = gcd(input_rate, SAMPLE_RATE)
        resampled = resample_poly(
            waveform.numpy(),
            SAMPLE_RATE // common_divisor,
            input_rate // common_divisor,
        )
        waveform = torch.as_tensor(resampled, dtype=torch.float32)
    if truncated and limit is not None:
        waveform = waveform[: round(limit.seconds * SAMPLE_RATE)]
    if waveform.numel() < MIN_SAMPLES:
        message = "Audio is too short after resampling."
        raise AudioInputError(message)
    return LoadedAudio(np.asarray(waveform, dtype=np.float32), truncated)
