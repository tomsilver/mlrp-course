"""Run online planning with expectimax search."""

from functools import lru_cache

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState


def expectimax_search(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    search_horizon: int,
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
        if h == search_horizon:
            return 0
        return max(Q(s, a, h) for a in A)

    @lru_cache(maxsize=None)
    def Q(s, a, h):
        """Shorthand for the action-value function."""
        return sum(P(s, a)[ns] * (R(s, a, ns) + gamma * V(ns, h + 1)) for ns in P(s, a))

    return max(A, key=lambda a: Q(initial_state, a, 0))