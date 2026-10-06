import numpy as np
import pytest

from komos.audio import SAMPLE_RATE, load_audio


def test_load_audio_resamples_waveform_to_model_rate() -> None:
    input_rate = 24000
    duration = 0.25
    sample_count = round(input_rate * duration)
    time = np.arange(sample_count, dtype=np.float32) / input_rate
    waveform = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    loaded = load_audio(waveform, input_rate)

    assert loaded.waveform.dtype == np.float32
    assert len(loaded.waveform) == round(SAMPLE_RATE * duration)
    frequencies = np.fft.rfftfreq(len(loaded.waveform), 1 / SAMPLE_RATE)
    peak_frequency = frequencies[np.argmax(np.abs(np.fft.rfft(loaded.waveform)))]
    assert peak_frequency == pytest.approx(440, abs=4)
