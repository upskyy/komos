import numpy as np
from numpy.typing import NDArray

def resample_poly(
    x: NDArray[np.float32],
    up: int,
    down: int,
    axis: int = 0,
    window: tuple[str, float] = ("kaiser", 5.0),
    padtype: str = "constant",
    cval: float | None = None,
) -> NDArray[np.float32]: ...
