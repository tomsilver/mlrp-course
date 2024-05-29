"""A generic definition of a POMDP with discrete spaces."""

import abc
from typing import Generic, Set, TypeAlias, TypeVar

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import CategoricalDistribution, HashableComparable

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
    ) -> CategoricalDistribution[_O]:
        """Return a discrete distribution over observations."""

    @abc.abstractmethod
    def get_initial_observation_distribution(
        self, initial_state: _S
    ) -> CategoricalDistribution[_O]:
        """Return a discrete distribution over observations."""

    def sample_observation(
        self, action: _A, next_state: _S, rng: np.random.Generator
    ) -> _O:
        """Sample an observation from the observation distribution."""
        return self.get_observation_distribution(action, next_state).sample(rng)

    def sample_initial_observation(
        self, initial_state: _S, rng: np.random.Generator
    ) -> _O:
        """Sample an initial observation."""
        return self.get_initial_observation_distribution(initial_state).sample(rng)

    def get_observation_probability(self, action: _A, next_state: _S, obs: _O) -> float:
        """Convenience method for some algorithms."""
        return self.get_observation_distribution(action, next_state)(obs)


class BeliefState(CategoricalDistribution[DiscreteState]):
    """A belief state for a DiscretePOMDP."""
