import warnings

import numpy as np
import pytest
import torch

from komos import KOMOS, KOMOSConfig
from komos.audio import AudioTruncationWarning


@pytest.fixture(scope="module")
def model() -> KOMOS:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = KOMOSConfig(
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        max_source_positions=500,
        listener_emb_dim=2,
        blstm_hidden=4,
        calib_a=1.2,
        calib_b=3.0,
    )
    return KOMOS(config).eval()


def test_default_uses_first_ten_seconds(model: KOMOS) -> None:
    wave = np.random.default_rng(42).normal(0, 0.1, 176000).astype(np.float32)
    expected = model.score(wave[:160000], 16000)
    with pytest.warns(AudioTruncationWarning):
        actual = model.score(wave, 16000)
    assert actual == pytest.approx(expected, abs=1e-7)


def test_ten_seconds_emits_no_warning(model: KOMOS) -> None:
    with warnings.catch_warnings(record=True) as caught:
        _ = model.score(np.zeros(160000, dtype=np.float32), 16000)
    assert not caught


def test_long_form_weights_partial_window_by_duration(model: KOMOS) -> None:
    wave = np.random.default_rng(43).normal(0, 0.1, 208000).astype(np.float32)
    first = model.score(wave[:160000], 16000)
    last = model.score(wave[160000:], 16000)
    expected = (10 * first + 3 * last) / 13
    actual = model.score(wave, 16000, long_form=True)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_long_form_matches_short_input(model: KOMOS) -> None:
    wave = np.zeros(16000, dtype=np.float32)
    expected = model.score(wave, 16000)
    actual = model.score(wave, 16000, long_form=True)
    assert actual == pytest.approx(expected, abs=1e-7)


def test_long_form_handles_single_sample_tail(model: KOMOS) -> None:
    actual = model.score(np.zeros(160001, dtype=np.float32), 16000, long_form=True)
    assert 1 <= actual <= 5


def test_score_restores_training_state(model: KOMOS) -> None:
    _ = model.train()
    _ = model.score(np.zeros(16000, dtype=np.float32), 16000)
    assert model.training
    _ = model.eval()


def test_long_form_clips_only_after_aggregation(
    model: KOMOS, monkeypatch: pytest.MonkeyPatch
) -> None:
    wave = np.random.default_rng(44).normal(0, 0.1, 208000).astype(np.float32)
    first = model.score(wave[:160000], 16000)
    last = model.score(wave[160000:], 16000)
    assert 1 < first < 5
    assert 1 < last < 5
    raw_first = (first - model.mos_config.calib_b) / model.mos_config.calib_a
    raw_last = (last - model.mos_config.calib_b) / model.mos_config.calib_a
    a = 6 / abs(raw_last - raw_first)
    b = -a * min(raw_first, raw_last)
    expected = min(5, max(1, a * (10 * raw_first + 3 * raw_last) / 13 + b))
    monkeypatch.setattr(model.mos_config, "calib_a", a)
    monkeypatch.setattr(model.mos_config, "calib_b", b)
    actual = model.score(wave, 16000, long_form=True)
    assert actual == pytest.approx(expected, abs=1e-5)
