from os import PathLike

import numpy as np
from numpy.typing import NDArray

class SoundFileInfo:
    frames: int
    samplerate: int

def info(file: str | PathLike[str]) -> SoundFileInfo: ...
def read(
    file: str | PathLike[str],
    frames: int = -1,
    *,
    dtype: str = "float64",
    always_2d: bool = False,
) -> tuple[NDArray[np.float32], int]: ...
def write(
    file: str | PathLike[str],
    data: NDArray[np.float32],
    samplerate: int,
    subtype: str | None = None,
) -> None: ...
