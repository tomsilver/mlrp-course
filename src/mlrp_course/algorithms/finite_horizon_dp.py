"""Solve finite-horizon MDPs by dynamic programming."""

from typing import Dict

from mlrp_course.mdp.discrete_mdp import DiscreteMDP, DiscreteState


def finite_horizon_dp(
    mdp: DiscreteMDP,
) -> Dict[int, Dict[DiscreteState, float]]:
    """Solve finite-horizon MDPs by dynamic programming."""
    assert mdp.horizon is not None

    # Get states, actions, P, and R.
    S = mdp.state_space
    A = mdp.action_space
    gamma = mdp.temporal_discount_factor
    P = mdp.get_transition_probability
    R = mdp.get_reward

    V: Dict[int, Dict[DiscreteState, float]] = {}

    # Base case: all values are zero at the end.
    V[mdp.horizon] = {s: 0.0 for s in S}

    for t in range(mdp.horizon - 1, -1, -1):
        V[t] = {}
        for s in S:
            V[t][s] = max(
                sum(P(s, a, ns) * (R(s, a, ns) + gamma * V[t + 1][ns]) for ns in S)
                for a in A
            )

    return V
