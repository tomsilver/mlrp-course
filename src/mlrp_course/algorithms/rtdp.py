"""Real-time dynamic programming."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import AlgorithmConfig
from mlrp_course.utils import (
    bellman_backup,
    sample_trajectory,
    value_function_to_greedy_policy,
)


@dataclass(frozen=True)
class RTDPConfig(AlgorithmConfig):
    """Hyperparameters for RTDP."""

    search_horizon: int = 10
    num_trajectory_samples: int = 10


def rtdp(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    rng: np.random.Generator,
    config: RTDPConfig,
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


def get_policy_rtdp(
    state: DiscreteState,
    mdp: DiscreteMDP,
    rng: np.random.Generator,
    config: AlgorithmConfig,
) -> Callable[[DiscreteState], DiscreteAction]:
    """Create a policy that runs RTDP internally."""
    assert isinstance(config, RTDPConfig)
    del state  # Instead of planning once, re-plan at every state

    def pi(s: DiscreteState) -> DiscreteAction:
        """Run RTDP on every step."""
        return rtdp(s, mdp, rng, config)

    return pi
