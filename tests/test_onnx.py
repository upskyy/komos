from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from komos import KOMOS, KOMOSConfig
from komos.onnx import KOMOSOnnx


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> tuple[KOMOS, KOMOSOnnx, Path]:
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
    torch_model = KOMOS(config).eval()
    model_path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    torch.onnx.export(
        torch_model,
        (
            torch.zeros((1, 80, 1000), dtype=torch.float32),
            torch.ones((1, 160000), dtype=torch.int64),
        ),
        model_path,
        input_names=["input_features", "attention_mask"],
        output_names=["raw_score"],
        opset_version=17,
        dynamo=False,
        external_data=False,
        dynamic_axes={
            "input_features": {0: "batch"},
            "attention_mask": {0: "batch"},
            "raw_score": {0: "batch"},
        },
    )
    onnx_model = onnx.load(model_path)
    onnx.helper.set_model_props(
        onnx_model,
        {
            "num_mel_bins": str(config.num_mel_bins),
            "sample_rate": str(config.sample_rate),
            "max_seconds": str(config.max_seconds),
            "calib_a": str(config.calib_a),
            "calib_b": str(config.calib_b),
            "score_clip_min": str(config.score_clip_min),
            "score_clip_max": str(config.score_clip_max),
        },
    )
    onnx.save(onnx_model, model_path)
    return torch_model, KOMOSOnnx(model_path), model_path


def test_onnx_score_matches_torch(models: tuple[KOMOS, KOMOSOnnx, Path]) -> None:
    torch_model, onnx_model, _ = models
    waveform = np.random.default_rng(42).normal(0, 0.1, 80000).astype(np.float32)
    expected = torch_model.score(waveform, 16000)
    actual = onnx_model.score(waveform, 16000)
    assert actual == pytest.approx(expected, abs=1e-5)


def test_onnx_graph_supports_dynamic_batch(models: tuple[KOMOS, KOMOSOnnx, Path]) -> None:
    _, _, model_path = models
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    output = session.run(
        ["raw_score"],
        {
            "input_features": np.zeros((2, 80, 1000), dtype=np.float32),
            "attention_mask": np.ones((2, 160000), dtype=np.int64),
        },
    )[0]
    onnx_model = onnx.load(model_path)
    assert onnx_model.opset_import[0].version == 17
    assert inputs[0].shape == ["batch", 80, 1000]
    assert inputs[1].shape == ["batch", 160000]
    assert np.asarray(output).shape == (2,)


def test_onnx_long_form_matches_torch(models: tuple[KOMOS, KOMOSOnnx, Path]) -> None:
    torch_model, onnx_model, _ = models
    waveform = np.random.default_rng(43).normal(0, 0.1, 208000).astype(np.float32)
    expected = torch_model.score_long_form(waveform, 16000)
    actual = onnx_model.score_long_form(waveform, 16000)
    assert actual == pytest.approx(expected, abs=1e-5)
