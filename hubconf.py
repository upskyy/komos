import sys
from pathlib import Path

import torch

dependencies = [
    "torch",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "numpy",
    "scipy",
    "soundfile",
    "typing_extensions",
]


def komos(
    model_id: str = "upskyy/komos", device: str = "cpu", revision: str = "v0.0.2"
) -> torch.nn.Module:
    """Load KOMOS weights from Hugging Face."""
    source_root = str(Path(__file__).resolve().parent / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from komos import KOMOS  # noqa: PLC0415

    model = KOMOS.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        model_id, revision=revision, dtype=torch.float32
    )

    # Transformers' decorated to() has an unbound-self typing defect.
    _ = model.to(device=device)  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]

    return model.eval()
