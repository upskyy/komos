import torch

from komos import KOMOS

dependencies = ["komos", "torch", "transformers", "numpy", "scipy", "soundfile"]


def komos(model_id: str = "upskyy/komos", device: str = "cpu", revision: str = "main") -> KOMOS:
    """Load KOMOS weights from Hugging Face."""
    model = KOMOS.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        model_id, revision=revision, dtype=torch.float32
    )
    # Transformers' decorated to() has an unbound-self typing defect.
    _ = model.to(device=device)  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]
    return model.eval()
