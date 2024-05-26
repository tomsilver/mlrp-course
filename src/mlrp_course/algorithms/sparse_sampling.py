"""Sparse sampling."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import AlgorithmConfig


@dataclass(frozen=True)
class SparseSamplingConfig(AlgorithmConfig):
    """Hyperparameters for sparse sampling."""

    search_horizon: int = 10
    num_samples: int = 5


def sparse_sampling(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    rng: np.random.Generator,
    config: SparseSamplingConfig,
) -> DiscreteAction:
    """Returns a single action to take."""
    # Note: no iteration over state space.
    A = mdp.action_space
    R = mdp.get_reward
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
        qsa = 0.0
        for _ in range(config.num_samples):
            ns = mdp.sample_next_state(s, a, rng)
            qsa += 1 / config.num_samples * (R(s, a, ns) + gamma * V(ns, h + 1))
        return qsa

    return max(A, key=lambda a: Q(initial_state, a, 0))


def get_policy_sparse_sampling(
    state: DiscreteState,
    mdp: DiscreteMDP,
    rng: np.random.Generator,
    config: AlgorithmConfig,
) -> Callable[[DiscreteState], DiscreteAction]:
    """Create a policy that runs sparse sampling internally."""
    del state  # Instead of planning once, re-plan at every state
    assert isinstance(config, SparseSamplingConfig)

    def pi(s: DiscreteState) -> DiscreteAction:
        """Run sparse sampling on every step."""
        return sparse_sampling(s, mdp, rng, config)

    return pi
