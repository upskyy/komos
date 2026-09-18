import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from typing_extensions import override

from .audio import AudioInput, load_audio
from .configuration_komos import KomosConfig


class KomosForMOS(PreTrainedModel):
    config_class: type[PretrainedConfig] | None = KomosConfig
    base_model_prefix: str = "backbone"
    main_input_name: str = "input_features"
    _supports_sdpa: bool = True

    def __init__(self, config: KomosConfig) -> None:
        super().__init__(config)  # pyright: ignore[reportUnknownMemberType]
        self.mos_config: KomosConfig = config
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
        waveform = load_audio(audio, sr)
        window_size = round(self.mos_config.max_seconds * self.mos_config.sample_rate)
        if len(waveform) <= window_size:
            windows = [waveform]
        else:
            hop = window_size // 2
            windows = [
                waveform[start : start + window_size]
                for start in range(0, len(waveform) - window_size + 1, hop)
            ]
            if (len(waveform) - window_size) % hop:
                windows.append(waveform[-window_size:])
        was_training = self.training
        _ = self.eval()
        try:
            predictions: list[float] = []
            for window in windows:
                inputs = self.feature_extractor(
                    window, sampling_rate=self.mos_config.sample_rate, return_tensors="pt"
                )
                features = inputs["input_features"]  # pyright: ignore[reportAny]
                assert isinstance(features, torch.Tensor)
                features = features.to(device=self.device, dtype=self.dtype)
                mask = torch.ones((1, len(window)), device=self.device, dtype=torch.long)
                predictions.append(self.forward(features, mask).item())
            calibrated = (
                self.mos_config.calib_a * float(np.mean(predictions)) + self.mos_config.calib_b
            )
            return min(
                self.mos_config.score_clip_max, max(self.mos_config.score_clip_min, calibrated)
            )
        finally:
            _ = self.train(was_training)
