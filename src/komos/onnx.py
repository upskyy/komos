import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from transformers import WhisperFeatureExtractor
from typing_extensions import Self

from komos.audio import AudioInput, AudioTruncationWarning, load_audio
from komos.config import KOMOSConfig


class _OnnxConfigError(ValueError):
    pass


def _load_config(metadata: Mapping[str, str]) -> KOMOSConfig:
    try:
        return KOMOSConfig(
            num_mel_bins=int(metadata["num_mel_bins"]),
            sample_rate=int(metadata["sample_rate"]),
            max_seconds=float(metadata["max_seconds"]),
            calib_a=float(metadata["calib_a"]),
            calib_b=float(metadata["calib_b"]),
            score_clip_min=float(metadata["score_clip_min"]),
            score_clip_max=float(metadata["score_clip_max"]),
        )
    except (KeyError, ValueError) as error:
        raise _OnnxConfigError(str(error)) from error


class _WhisperFeatureExtractor(WhisperFeatureExtractor):
    def extract(self, waveform_batch: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.asarray(self._np_extract_fbank_features(waveform_batch, "cpu"), dtype=np.float32)


class KOMOSOnnx:
    """KOMOS inference backed by ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.session: ort.InferenceSession = ort.InferenceSession(
            str(model_path), providers=list(providers or ["CPUExecutionProvider"])
        )
        self.config: KOMOSConfig = _load_config(self.session.get_modelmeta().custom_metadata_map)
        self.feature_extractor: _WhisperFeatureExtractor = _WhisperFeatureExtractor(
            feature_size=self.config.num_mel_bins,
            sampling_rate=self.config.sample_rate,
            chunk_length=int(self.config.max_seconds),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "upskyy/komos",
        *,
        revision: str = "main",
        providers: Sequence[str] | None = None,
    ) -> Self:
        """Load an ONNX model and its configuration from Hugging Face."""
        model_path = hf_hub_download(model_id, "model.onnx", revision=revision)
        return cls(model_path, providers)

    def score(self, audio: AudioInput, sr: int | None = None) -> float:
        """Return the utterance-level KOMOS score using up to the first 10 seconds."""
        waveform = load_audio(audio, sr)
        window_size = round(self.config.max_seconds * self.config.sample_rate)
        if len(waveform) > window_size:
            limit = self.config.max_seconds
            warnings.warn(
                f"Scoring only the first {limit:g}s. Use score_long_form() to score all chunks.",
                AudioTruncationWarning,
                stacklevel=3,
            )
        return self._score_windows([waveform[:window_size]])

    def score_long_form(self, audio: AudioInput, sr: int | None = None) -> float:
        """Return a duration-weighted score for non-overlapping audio chunks."""
        waveform = load_audio(audio, sr)
        window_size = round(self.config.max_seconds * self.config.sample_rate)
        windows = [
            waveform[start : start + window_size] for start in range(0, len(waveform), window_size)
        ]
        return self._score_windows(windows)

    def _score_windows(self, windows: Sequence[NDArray[np.float32]]) -> float:
        window_size = round(self.config.max_seconds * self.config.sample_rate)
        waveform_batch = np.zeros((len(windows), window_size), dtype=np.float32)
        attention_mask = np.zeros((len(windows), window_size), dtype=np.int64)
        durations = np.empty(len(windows), dtype=np.int64)
        for index, window in enumerate(windows):
            duration = len(window)
            waveform_batch[index, :duration] = window
            attention_mask[index, :duration] = 1
            durations[index] = duration
        features = self.feature_extractor.extract(waveform_batch)
        outputs = self.session.run(
            ["raw_score"],
            {
                "input_features": features,
                "attention_mask": attention_mask,
            },
        )
        raw_scores = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        weighted_score = float(np.average(raw_scores, weights=durations))
        calibrated = self.config.calib_a * weighted_score + self.config.calib_b
        return min(self.config.score_clip_max, max(self.config.score_clip_min, calibrated))
