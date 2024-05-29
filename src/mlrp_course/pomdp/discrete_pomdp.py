"""A generic definition of a POMDP with discrete spaces."""

import abc
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, Set, Tuple, TypeAlias, TypeVar

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import HashableComparable
from mlrp_course.utils import sample_from_categorical

DiscreteObs: TypeAlias = HashableComparable

_O = TypeVar("_O", bound=DiscreteObs)
_S = TypeVar("_S", bound=DiscreteState)
_A = TypeVar("_A", bound=DiscreteAction)


class DiscretePOMDP(Generic[_O, _S, _A], DiscreteMDP[_S, _A]):
    """A Partially Observable Markov Decision Process."""

    @property
    @abc.abstractmethod
    def observation_space(self) -> Set[_O]:
        """Representation of the POMDP observation set."""

    @abc.abstractmethod
    def get_observation_distribution(
        self, action: _A, next_state: _S
    ) -> Dict[_O, float]:
        """Return a discrete distribution over observations."""

    @abc.abstractmethod
    def get_initial_observation_distribution(
        self, initial_state: _S
    ) -> Dict[_O, float]:
        """Return a discrete distribution over observations."""

    def sample_observation(
        self, action: _A, next_state: _S, rng: np.random.Generator
    ) -> _O:
        """Sample an observation from the observation distribution."""
        dist = self.get_observation_distribution(action, next_state)
        return sample_from_categorical(dist, rng)

    def sample_initial_observation(
        self, initial_state: _S, rng: np.random.Generator
    ) -> _O:
        """Sample an initial observation."""
        dist = self.get_initial_observation_distribution(initial_state)
        return sample_from_categorical(dist, rng)

    def get_observation_probability(self, action: _A, next_state: _S, obs: _O) -> float:
        """Convenience method for some algorithms."""
        return self.get_observation_distribution(action, next_state).get(obs, 0.0)


@dataclass(frozen=True)
class BeliefState:
    """A belief state for a DiscretePOMDP."""

    state_to_prob: Dict[DiscreteState, float]

    def __post_init__(self) -> None:
        assert np.isclose(sum(self.state_to_prob.values()), 1.0)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.state_to_prob.items())))

    def __call__(self, state: DiscreteState) -> float:
        """Get the probability for the given state."""
        return self.state_to_prob.get(state, 0.0)

    def __iter__(self) -> Iterator[DiscreteState]:
        return iter(self.state_to_prob)

    def __str__(self) -> str:
        return str(sorted(self.items()))

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, BeliefState)
        states = set(self) | set(other)
        return all(np.isclose(self(s), other(s)) for s in states)

    def __lt__(self, other: Any) -> bool:
        assert isinstance(other, BeliefState)
        return str(self) < str(other)

    def items(self) -> Iterator[Tuple[DiscreteState, float]]:
        """Iterate the dictionary."""
        return iter(self.state_to_prob.items())
