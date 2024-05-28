"""Data structures."""

from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Type aliases.
Image: TypeAlias = NDArray[np.uint8]


class Hyperparameters:
    """General hyperparameters."""


_T = TypeVar("_T", bound="HashableComparable")


@runtime_checkable
class HashableComparable(Protocol):
    """Used for type checking objects that must be hashable and comparable."""

    def __hash__(self) -> int: ...

    def __eq__(self, other: Any) -> bool: ...

    def __lt__(self: _T, other: _T) -> bool: ...
