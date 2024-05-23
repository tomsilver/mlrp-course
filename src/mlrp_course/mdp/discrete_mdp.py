"""A generic definition of an MDP with discrete states and actions."""

import abc
from typing import Dict, Generic, Hashable, Set, TypeVar

import numpy as np

from mlrp_course.structs import Image

State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action")


class DiscreteMDP(Generic[State, Action]):
    """A Markov Decision Process."""

    @property
    @abc.abstractmethod
    def state_space(self) -> Set[State]:
        """Representation of the MDP state set."""

    @property
    @abc.abstractmethod
    def action_space(self) -> Set[Action]:
        """Representation of the MDP action set."""

    @property
    def temporal_discount_factor(self) -> float:
        """Gamma, defaults to 1."""
        return 1.0

    @property
    def horizon(self) -> float:
        """H, defaults to inf."""
        return float("inf")

    @abc.abstractmethod
    def state_is_terminal(self, state: State) -> bool:
        """Designate certain states as terminal (done) states."""

    @abc.abstractmethod
    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        """Return (deterministic) reward for executing action in state."""

    @abc.abstractmethod
    def get_transition_distribution(
        self, state: State, action: Action
    ) -> Dict[State, float]:
        """Return a discrete distribution over next states."""

    def sample_next_state(
        self, state: State, action: Action, rng: np.random.Generator
    ) -> State:
        """Sample a next state from the transition distribution.

        This function may be overwritten by subclasses when the explicit
        distribution is too large to enumerate.
        """
        next_state_dist = self.get_transition_distribution(state, action)
        next_states, probs = zip(*next_state_dist.items(), strict=True)
        next_state_index = rng.choice(len(next_states), p=probs)
        next_state = next_states[next_state_index]
        return next_state

    @abc.abstractmethod
    def render_state(self, state: State) -> Image:
        """Optional rendering function for visualizations."""
