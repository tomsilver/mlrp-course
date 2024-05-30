"""Data structures."""

from functools import cached_property
from typing import (
    Any,
    Dict,
    Generic,
    Hashable,
    Iterator,
    Protocol,
    Tuple,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

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


class CategoricalDistribution(Generic[_T]):
    """A categorical distribution."""

    def __init__(
        self, outcome_to_prob: Dict[_T, float], normalize: bool = False
    ) -> None:
        # Prune zero entries and avoid modifying input.
        d = {o: p for o, p in outcome_to_prob.items() if p > 0}
        # Normalize the distribution if asked.
        if normalize:
            z = sum(d.values())
            assert z > 0, "Categorical distribution has no nonzero probability"
            d = {o: p / z for o, p in d.items()}
        # Validate the distribution.
        assert np.isclose(sum(d.values()), 1.0)
        # Finalize.
        self._outcome_to_prob = d

    @cached_property
    def _hashable(self) -> Hashable:
        # NOTE: it is important to sort here, even though dictionary ordering
        # is deterministic, because the same distribution may be created with
        # two different dictionary orderings.
        return tuple(sorted(self._outcome_to_prob.items()))

    @cached_property
    def _hash(self) -> int:
        return hash(self._hashable)

    @cached_property
    def _str(self) -> str:
        return f"CategoricalDistribution({self._hashable})"

    def __hash__(self) -> int:
        return self._hash

    def __call__(self, outcome: _T) -> float:
        """Get the probability for the given outcome."""
        return self[outcome]

    def __getitem__(self, outcome: _T) -> float:
        return self._outcome_to_prob.get(outcome, 0.0)

    def __iter__(self) -> Iterator[_T]:
        return iter(self._outcome_to_prob)

    def __repr__(self) -> str:
        return self._str

    def __str__(self) -> str:
        return self._str

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, CategoricalDistribution)
        outcomes = set(self) | set(other)
        return all(np.isclose(self(o), other(o)) for o in outcomes)

    def __lt__(self, other: Any) -> bool:
        assert isinstance(other, CategoricalDistribution)
        return str(self) < str(other)

    def items(self) -> Iterator[Tuple[_T, float]]:
        """Iterate the dictionary."""
        return iter(self._outcome_to_prob.items())

    def sample(self, rng: np.random.Generator) -> _T:
        """Draw a random sample."""
        candidates, probs = zip(*self._outcome_to_prob.items(), strict=True)
        idx = rng.choice(len(candidates), p=probs)
        return candidates[idx]
