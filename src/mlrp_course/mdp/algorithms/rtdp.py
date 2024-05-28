"""Real-time dynamic programming."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import numpy as np

from mlrp_course.agents import DiscreteMDPAgent
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import Hyperparameters
from mlrp_course.utils import (
    bellman_backup,
    sample_trajectory,
    value_function_to_greedy_policy,
)


@dataclass(frozen=True)
class RTDPHyperparameters(Hyperparameters):
    """Hyperparameters for RTDP."""

    search_horizon: int = 10
    num_trajectory_samples: int = 10


def rtdp(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    rng: np.random.Generator,
    config: RTDPHyperparameters,
) -> DiscreteAction:
    """Real-time dynamic programming."""

    # Lazily initialize value function.
    V: Dict[DiscreteState, float] = defaultdict(float)

    for _ in range(config.num_trajectory_samples):
        # Turn value function estimate into greedy policy.
        pi = value_function_to_greedy_policy(V, mdp, rng)
        # Collect a trajectory.
        states, _ = sample_trajectory(
            initial_state, pi, mdp, config.search_horizon, rng
        )
        # Update the values.
        for s in states[::-1]:
            V[s] = bellman_backup(s, V, mdp)

    # Finally, return an action.
    pi = value_function_to_greedy_policy(V, mdp, rng)
    return pi(initial_state)


class RTDPAgent(DiscreteMDPAgent):
    """An agent that runs RTDP at every timestep."""

    def __init__(self, planner_config: RTDPHyperparameters, *args, **kwargs) -> None:
        self._planner_config = planner_config
        super().__init__(*args, **kwargs)

    def _get_action(self) -> DiscreteAction:
        assert self._last_observation is not None
        return rtdp(self._last_observation, self._mdp, self._rng, self._planner_config)
