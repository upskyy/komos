from dataclasses import dataclass
from typing import ClassVar

from transformers import WhisperConfig


@dataclass
class KOMOSConfig(WhisperConfig):
    model_type: ClassVar[str] = "komos"
    listener_emb_dim: int = 128
    num_listeners: int = 31
    mean_token_id: int = 29
    blstm_hidden: int = 256
    blstm_layers: int = 1
    max_source_positions: int = 500
    sample_rate: int = 16000
    max_seconds: float = 10.0
    calib_a: float = 1.0
    calib_b: float = 0.0
    score_clip_min: float = 1.0
    score_clip_max: float = 5.0
