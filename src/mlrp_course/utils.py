"""Utilities."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteMDP, DiscreteState
from mlrp_course.structs import Image


def bellman_backup(
    s: DiscreteState, V: Dict[DiscreteState, float], mdp: DiscreteMDP
) -> float:
    """Look ahead one step and propose an update for the value of s."""
    assert mdp.horizon == float("inf")
    vs = -float("inf")
    for a in mdp.action_space:
        qsa = 0.0
        for ns, p in mdp.get_transition_distribution(s, a).items():
            r = mdp.get_reward(s, a, ns)
            qsa += p * (r + mdp.temporal_discount_factor * V[ns])
        vs = max(qsa, vs)
    return vs


def fig2data(fig: plt.Figure) -> Image:
    """Convert matplotlib figure into Image."""
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())
