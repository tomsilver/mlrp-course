"""A generic definition of a POMDP with discrete spaces."""

import abc
from typing import Dict, Generic, Set, TypeAlias, TypeVar

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import HashableComparable

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
    def get_observation_distribution(self, state: _S, action: _A) -> Dict[_O, float]:
        """Return a discrete distribution over observations."""
