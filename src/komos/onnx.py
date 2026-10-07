import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from transformers import WhisperFeatureExtractor
from typing_extensions import Self

from komos.audio import (
    MAX_LONG_FORM_SECONDS,
    AudioInput,
    AudioLimit,
    AudioTruncationWarning,
    load_audio,
)
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
        inputs = self(
            waveform_batch,
            sampling_rate=self.sampling_rate,
            return_tensors="np",
        )
        features = cast("NDArray[np.float32]", inputs["input_features"])
        return np.asarray(features, dtype=np.float32)


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
        loaded = load_audio(
            audio,
            sr,
            AudioLimit(self.config.max_seconds, truncate=True),
        )

        if loaded.truncated:
            limit = self.config.max_seconds
            warnings.warn(
                f"Scoring only the first {limit:g}s. Use score_long_form() to score all chunks.",
                AudioTruncationWarning,
                stacklevel=3,
            )

        return self._score_windows([loaded.waveform])

    def score_long_form(self, audio: AudioInput, sr: int | None = None) -> float:
        """Score up to one minute using duration-weighted, non-overlapping chunks."""
        loaded = load_audio(
            audio,
            sr,
            AudioLimit(MAX_LONG_FORM_SECONDS, truncate=False),
        )

        waveform = loaded.waveform
        window_size = round(self.config.max_seconds * self.config.sample_rate)
        windows = [
            waveform[start : start + window_size] for start in range(0, len(waveform), window_size)
        ]

        return self._score_windows(windows)

    def _score_windows(self, windows: Sequence[NDArray[np.float32]]) -> float:
        window_size = round(self.config.max_seconds * self.config.sample_rate)
        weighted_score = 0.0
        total_samples = 0

        for window in windows:
            duration = len(window)
            waveform_batch = np.zeros((1, window_size), dtype=np.float32)
            attention_mask = np.zeros((1, window_size), dtype=np.int64)

            waveform_batch[0, :duration] = window
            attention_mask[0, :duration] = 1

            features = self.feature_extractor.extract(waveform_batch)
            outputs = self.session.run(
                ["raw_score"],
                {
                    "input_features": features,
                    "attention_mask": attention_mask,
                },
            )
            raw_score = float(np.average(outputs[0]))

            weighted_score += raw_score * duration
            total_samples += duration

        calibrated = self.config.calib_a * (weighted_score / total_samples) + self.config.calib_b

        return min(self.config.score_clip_max, max(self.config.score_clip_min, calibrated))
