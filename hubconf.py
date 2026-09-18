from komos import KomosForMOS

dependencies = ["torch", "torchaudio", "transformers", "numpy", "soundfile", "huggingface_hub"]


def komos(
    model_id: str = "upskyy/komos", device: str = "cpu", revision: str = "main"
) -> KomosForMOS:
    """Load KOMOS weights from Hugging Face."""
    model = KomosForMOS.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        model_id, revision=revision
    )
    # Transformers' decorated to() has an unbound-self typing defect.
    _ = model.to(device=device)  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]
    return model.eval()
