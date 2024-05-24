"""Utilities."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import Image


def bellman_backup(
    s: DiscreteState, V: Dict[DiscreteState, float], mdp: DiscreteMDP
) -> float:
    """Look ahead one step and propose an update for the value of s."""
    assert mdp.horizon is None
    vs = -float("inf")
    for a in mdp.action_space:
        qsa = 0.0
        for ns, p in mdp.get_transition_distribution(s, a).items():
            r = mdp.get_reward(s, a, ns)
            qsa += p * (r + mdp.temporal_discount_factor * V[ns])
        vs = max(qsa, vs)
    return vs


def value_to_action_value_function(
    V: Dict[DiscreteState, float], mdp: DiscreteMDP
) -> Dict[DiscreteState, Dict[DiscreteAction, float]]:
    """Convert a value (V) function to an action-value (Q) function."""
    Q: Dict[DiscreteState, Dict[DiscreteAction, float]] = {}

    # Get states, P, and R.
    S = mdp.state_space
    A = mdp.action_space
    gamma = mdp.temporal_discount_factor
    P = mdp.get_transition_probability
    R = mdp.get_reward

    for s in S:
        Q[s] = {}
        for a in A:
            Q[s][a] = sum(P(s, a, ns) * (R(s, a, ns) * gamma * V[ns]) for ns in S)

    return Q


def fig2data(fig: plt.Figure) -> Image:
    """Convert matplotlib figure into Image."""
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())
