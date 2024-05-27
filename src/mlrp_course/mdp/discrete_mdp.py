"""A generic definition of an MDP with discrete states and actions."""

import abc
from typing import Dict, Generic, Optional, Set, TypeAlias, TypeVar

import numpy as np

from mlrp_course.structs import HashableComparable, Image

DiscreteState: TypeAlias = HashableComparable
DiscreteAction: TypeAlias = HashableComparable

_S = TypeVar("_S", bound=DiscreteState)
_A = TypeVar("_A", bound=DiscreteAction)


class DiscreteMDP(Generic[_S, _A]):
    """A Markov Decision Process."""

    @property
    @abc.abstractmethod
    def state_space(self) -> Set[_S]:
        """Representation of the MDP state set."""

    @property
    @abc.abstractmethod
    def action_space(self) -> Set[_A]:
        """Representation of the MDP action set."""

    @property
    def temporal_discount_factor(self) -> float:
        """Gamma, defaults to 1."""
        return 1.0

    @property
    def horizon(self) -> Optional[int]:
        """H, defaults to None (inf)."""
        return None

    @abc.abstractmethod
    def state_is_terminal(self, state: _S) -> bool:
        """Designate certain states as terminal (done) states."""

    @abc.abstractmethod
    def get_reward(self, state: _S, action: _A, next_state: _S) -> float:
        """Return (deterministic) reward for executing action in state."""

    @abc.abstractmethod
    def get_transition_distribution(self, state: _S, action: _A) -> Dict[_S, float]:
        """Return a discrete distribution over next states."""

    def sample_next_state(self, state: _S, action: _A, rng: np.random.Generator) -> _S:
        """Sample a next state from the transition distribution.

        This function may be overwritten by subclasses when the explicit
        distribution is too large to enumerate.
        """
        next_state_dist = self.get_transition_distribution(state, action)
        next_states, probs = zip(*next_state_dist.items(), strict=True)
        next_state_index = rng.choice(len(next_states), p=probs)
        next_state = next_states[next_state_index]
        return next_state

    def get_transition_probability(
        self, state: _S, action: _A, next_state: _S
    ) -> float:
        """Convenience method for some algorithms."""
        return self.get_transition_distribution(state, action).get(next_state, 0.0)

    @abc.abstractmethod
    def render_state(self, state: _S) -> Image:
        """Optional rendering function for visualizations."""
