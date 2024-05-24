"""Sparse sampling."""

from functools import lru_cache

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState


def sparse_sampling(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    search_horizon: int,
    rng: np.random.Generator,
    num_samples: int = 5,
) -> DiscreteAction:
    """Returns a single action to take."""
    # Note: no iteration over state space.
    A = mdp.action_space
    R = mdp.get_reward
    gamma = mdp.temporal_discount_factor

    @lru_cache(maxsize=None)
    def V(s, h):
        """Shorthand for the value function."""
        if h == search_horizon:
            return 0
        return max(Q(s, a, h) for a in A)

    @lru_cache(maxsize=None)
    def Q(s, a, h):
        """Shorthand for the action-value function."""
        qsa = 0.0
        for _ in range(num_samples):
            ns = mdp.sample_next_state(s, a, rng)
            qsa += 1 / num_samples * (R(s, a, ns) + gamma * V(ns, h + 1))
        return qsa

    return max(A, key=lambda a: Q(initial_state, a, 0))
