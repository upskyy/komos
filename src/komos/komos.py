import warnings
from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from typing_extensions import override

from komos.audio import (
    MAX_LONG_FORM_SECONDS,
    AudioInput,
    AudioLimit,
    AudioTruncationWarning,
    load_audio,
)
from komos.config import KOMOSConfig


class KOMOS(PreTrainedModel):
    config_class: type[PretrainedConfig] | None = KOMOSConfig
    base_model_prefix: str = "backbone"
    main_input_name: str = "input_features"
    _supports_sdpa: bool = True

    def __init__(self, config: KOMOSConfig) -> None:
        super().__init__(config)  # pyright: ignore[reportUnknownMemberType]
        self.mos_config: KOMOSConfig = config
        self.backbone: WhisperEncoder = WhisperEncoder(config)
        self.listener_embedding: nn.Embedding = nn.Embedding(
            config.num_listeners, config.listener_emb_dim
        )
        self.blstm: nn.LSTM = nn.LSTM(
            config.d_model + config.listener_emb_dim,
            config.blstm_hidden,
            num_layers=config.blstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj: nn.Linear = nn.Linear(2 * config.blstm_hidden, 1)
        self.feature_extractor: WhisperFeatureExtractor = WhisperFeatureExtractor(
            feature_size=config.num_mel_bins,
            sampling_rate=config.sample_rate,
            chunk_length=int(config.max_seconds),
        )
        self.post_init()

    @override
    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return raw utterance scores; attention_mask is in waveform samples."""
        hidden = self.backbone.forward(  # pyright: ignore[reportUnknownMemberType]
            input_features=input_features
        ).last_hidden_state
        assert isinstance(hidden, torch.Tensor)
        batch, frames, _ = hidden.shape
        if attention_mask is None:
            lengths = torch.full((batch,), frames, device=hidden.device, dtype=torch.long)
        else:
            lengths = (attention_mask.sum(-1) // 320).clamp(min=1, max=frames).long()
        listener_ids = torch.full(
            (batch,), self.mos_config.mean_token_id, device=hidden.device, dtype=torch.long
        )
        embedding = (
            self.listener_embedding.forward(listener_ids).unsqueeze(1).expand(-1, frames, -1)
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            torch.cat((hidden, embedding), dim=-1),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.blstm.forward(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=frames)
        frame_scores = self.proj.forward(padded).squeeze(-1)
        mask = torch.arange(frames, device=hidden.device).unsqueeze(0) < lengths.unsqueeze(1)
        return (frame_scores * mask).sum(-1) / lengths

    @torch.inference_mode()
    def score(self, audio: AudioInput, sr: int | None = None) -> float:
        """Return a calibrated MOS in [1, 5] for a file or mono waveform."""
        loaded = load_audio(
            audio,
            sr,
            AudioLimit(self.mos_config.max_seconds, truncate=True),
        )
        if loaded.truncated:
            limit = self.mos_config.max_seconds
            warnings.warn(
                f"Scoring only the first {limit:g}s. Use score_long_form() to score all chunks.",
                AudioTruncationWarning,
                stacklevel=3,
            )
        return self._score_windows([loaded.waveform])

    @torch.inference_mode()
    def score_long_form(self, audio: AudioInput, sr: int | None = None) -> float:
        """Score up to one minute using duration-weighted, non-overlapping chunks."""
        loaded = load_audio(
            audio,
            sr,
            AudioLimit(MAX_LONG_FORM_SECONDS, truncate=False),
        )
        waveform = loaded.waveform
        window_size = round(self.mos_config.max_seconds * self.mos_config.sample_rate)
        windows = [
            waveform[start : start + window_size] for start in range(0, len(waveform), window_size)
        ]
        return self._score_windows(windows)

    def _score_windows(self, windows: Sequence[NDArray[np.float32]]) -> float:
        _ = self.eval()
        weighted_score = 0.0
        total_samples = 0
        for window in windows:
            inputs = self.feature_extractor(
                window, sampling_rate=self.mos_config.sample_rate, return_tensors="pt"
            )
            features = inputs["input_features"]  # pyright: ignore[reportAny]
            assert isinstance(features, torch.Tensor)
            features = features.to(device=self.device, dtype=self.dtype)
            mask = torch.ones((1, len(window)), device=self.device, dtype=torch.long)
            weighted_score += self.forward(features, mask).item() * len(window)
            total_samples += len(window)
        calibrated = (
            self.mos_config.calib_a * (weighted_score / total_samples) + self.mos_config.calib_b
        )
        return min(self.mos_config.score_clip_max, max(self.mos_config.score_clip_min, calibrated))
