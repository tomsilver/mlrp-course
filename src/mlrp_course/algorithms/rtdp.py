"""Real-time dynamic programming."""

from collections import defaultdict
from typing import Dict

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.utils import (
    bellman_backup,
    sample_trajectory,
    value_function_to_greedy_policy,
)


def rtdp(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    search_horizon: int,
    rng: np.random.Generator,
    num_trajectory_samples: int = 10,
) -> DiscreteAction:
    """Real-time dynamic programming."""

    # Lazily initialize value function.
    V: Dict[DiscreteState, float] = defaultdict(float)

    for _ in range(num_trajectory_samples):
        # Turn value function estimate into greedy policy.
        pi = value_function_to_greedy_policy(V, mdp, rng)
        # Collect a trajectory.
        states, _ = sample_trajectory(initial_state, pi, mdp, search_horizon, rng)
        # Update the values.
        for s in states[::-1]:
            V[s] = bellman_backup(s, V, mdp)

    # Finally, return an action.
    pi = value_function_to_greedy_policy(V, mdp, rng)
    return pi(initial_state)
