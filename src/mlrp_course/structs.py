"""Data structures."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Type aliases.
Image: TypeAlias = NDArray[np.uint8]


class AlgorithmConfig:
    """General configuration for algorithms."""
