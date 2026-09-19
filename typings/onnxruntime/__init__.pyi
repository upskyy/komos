from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

InputArray: TypeAlias = NDArray[np.float32] | NDArray[np.int64]

class ModelMetadata:
    @property
    def custom_metadata_map(self) -> Mapping[str, str]: ...

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str,
        *,
        providers: Sequence[str] | None = None,
    ) -> None: ...
    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, InputArray],
        run_options: None = None,
    ) -> Sequence[NDArray[np.float32]]: ...
    def get_modelmeta(self) -> ModelMetadata: ...
