"""Value iteration."""

from dataclasses import dataclass
from typing import Dict, List

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.mdp.utils import (
    DiscreteMDPAgent,
    bellman_backup,
    value_function_to_greedy_policy,
)
from mlrp_course.structs import Hyperparameters


@dataclass(frozen=True)
class ValueIterationHyperparameters(Hyperparameters):
    """Hyperparameters for value iteration."""

    max_num_iterations: int = 1000
    change_threshold: float = 1e-4
    print_every: int | None = None


def value_iteration(
    mdp: DiscreteMDP,
    config: ValueIterationHyperparameters,
) -> List[Dict[DiscreteState, float]]:
    """Run value iteration for a certain number of iterations or until the max
    change between iterations is below a threshold.

    For analysis purposes, return the entire sequence of value function
    estimates.
    """
    # Initialize V to all zeros.
    V = {s: 0.0 for s in mdp.state_space}
    all_estimates = [V]

    for it in range(config.max_num_iterations):
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
        if config.print_every is not None and it % config.print_every == 0:
            print(f"VI max change after iteration {it} : {max_change}")

        if max_change < config.change_threshold:
            break

    return all_estimates


class ValueIterationAgent(DiscreteMDPAgent):
    """An agent that plans offline with value iteration."""

    def __init__(
        self,
        mdp: DiscreteMDP,
        seed: int,
        value_iteration_hyperparameters: ValueIterationHyperparameters | None = None,
    ) -> None:
        self._value_iteration_hyperparameters = (
            value_iteration_hyperparameters or ValueIterationHyperparameters()
        )
        super().__init__(mdp, seed)
        Vs = value_iteration(self._mdp, self._value_iteration_hyperparameters)
        self._pi = value_function_to_greedy_policy(Vs[-1], self._mdp, self._rng)

    def _get_action(self) -> DiscreteAction:
        assert self._last_observation is not None
        return self._pi(self._last_observation)
