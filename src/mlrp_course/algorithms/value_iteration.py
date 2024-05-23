"""Value iteration."""

from typing import Dict, List, Optional

from mlrp_course.mdp.discrete_mdp import DiscreteMDP, DiscreteState
from mlrp_course.utils import bellman_backup


def value_iteration(
    mdp: DiscreteMDP,
    max_num_iterations: int = 1000,
    change_threshold: float = 1e-4,
    print_every: Optional[int] = None,
) -> List[Dict[DiscreteState, float]]:
    """Run value iteration for a certain number of iterations or until the max
    change between iterations is below a threshold.

    For analysis purposes, return the entire sequence of value function
    estimates.
    """
    # Initialize V to all zeros.
    V = {s: 0.0 for s in mdp.state_space}
    all_estimates = [V]

    for it in range(max_num_iterations):
        next_V = {}
        max_change = 0.0
        for s in mdp.state_space:
            if mdp.state_is_terminal(s):
                next_V[s] = 0.0
            else:
                next_V[s] = bellman_backup(s, V, mdp)
            max_change = max(abs(next_V[s] - V[s]), max_change)

        V = next_V
        all_estimates.append(V)

        # Check if we can terminate early.
        if print_every is not None and it % print_every == 0:
            print(f"VI max change after iteration {it} : {max_change}")

        if max_change < change_threshold:
            break

    return all_estimates
