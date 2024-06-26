"""Real-time dynamic programming."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.mdp.utils import (
    DiscreteMDPAgent,
    bellman_backup,
    sample_trajectory,
    value_function_to_greedy_policy,
)
from mlrp_course.structs import Hyperparameters


@dataclass(frozen=True)
class RTDPHyperparameters(Hyperparameters):
    """Hyperparameters for RTDP."""

    max_search_horizon: int = 10
    num_trajectory_samples: int = 10


def rtdp(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    timestep: int,
    rng: np.random.Generator,
    config: RTDPHyperparameters,
) -> DiscreteAction:
    """Real-time dynamic programming."""

    # Calculate the remaining horizon.
    H = config.max_search_horizon
    if mdp.horizon is not None:
        H = min(H, mdp.horizon - timestep)

    # Lazily initialize value function.
    V: Dict[DiscreteState, float] = defaultdict(float)

    for _ in range(config.num_trajectory_samples):
        # Turn value function estimate into greedy policy.
        pi = value_function_to_greedy_policy(V, mdp, rng)
        # Collect a trajectory.
        states, _ = sample_trajectory(initial_state, pi, mdp, H, rng)
        # Update the values.
        for s in states[::-1]:
            V[s] = bellman_backup(s, V, mdp)

    # Finally, return an action.
    pi = value_function_to_greedy_policy(V, mdp, rng)
    return pi(initial_state)


class RTDPAgent(DiscreteMDPAgent):
    """An agent that runs RTDP at every timestep."""

    def __init__(
        self,
        mdp: DiscreteMDP,
        seed: int,
        rtdp_hyperparameters: RTDPHyperparameters | None = None,
    ) -> None:
        self._rtdp_hyperparameters = rtdp_hyperparameters or RTDPHyperparameters()
        super().__init__(mdp, seed)

    def _get_action(self) -> DiscreteAction:
        assert self._last_observation is not None
        return rtdp(
            self._last_observation,
            self._mdp,
            self._timestep,
            self._rng,
            self._rtdp_hyperparameters,
        )
