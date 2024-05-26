"""Run online planning with expectimax search."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import AlgorithmConfig


@dataclass(frozen=True)
class ExpectimaxSearchConfig(AlgorithmConfig):
    """Hyperparameters for expectimax search."""

    search_horizon: int = 10


def expectimax_search(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    config: ExpectimaxSearchConfig,
) -> DiscreteAction:
    """Returns a single action to take."""
    # Note: no iteration over state space.
    A = mdp.action_space
    R = mdp.get_reward
    P = mdp.get_transition_distribution
    gamma = mdp.temporal_discount_factor

    @lru_cache(maxsize=None)
    def V(s, h):
        """Shorthand for the value function."""
        if h == config.search_horizon:
            return 0
        return max(Q(s, a, h) for a in A)

    @lru_cache(maxsize=None)
    def Q(s, a, h):
        """Shorthand for the action-value function."""
        return sum(P(s, a)[ns] * (R(s, a, ns) + gamma * V(ns, h + 1)) for ns in P(s, a))

    return max(A, key=lambda a: Q(initial_state, a, 0))


def get_policy_expectimax_search(
    state: DiscreteState,
    mdp: DiscreteMDP,
    rng: np.random.Generator,
    config: AlgorithmConfig,
) -> Callable[[DiscreteState], DiscreteAction]:
    """Create a policy that runs expectimax search internally."""
    del rng  # not used
    del state  # Instead of planning once, re-plan at every state
    assert isinstance(config, ExpectimaxSearchConfig)

    def pi(s: DiscreteState) -> DiscreteAction:
        """Run expectimax search on every step."""
        return expectimax_search(s, mdp, config)

    return pi
