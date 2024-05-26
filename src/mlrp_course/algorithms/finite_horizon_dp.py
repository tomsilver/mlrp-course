"""Solve finite-horizon MDPs by dynamic programming."""

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import AlgorithmConfig
from mlrp_course.utils import value_function_to_greedy_policy


@dataclass(frozen=True)
class FiniteHorizonDPConfig(AlgorithmConfig):
    """Hyperparameters for finite-horizon DP."""


def finite_horizon_dp(
    mdp: DiscreteMDP,
    config: FiniteHorizonDPConfig,
) -> Dict[int, Dict[DiscreteState, float]]:
    """Solve finite-horizon MDPs by dynamic programming."""
    del config  # no hyperparameters used
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


def get_policy_finite_horizon_dp(
    mdp: DiscreteMDP, rng: np.random.Generator, config: AlgorithmConfig
) -> Callable[[DiscreteState], DiscreteAction]:
    """Run finite-horizon DP and produce a policy."""
    assert isinstance(config, FiniteHorizonDPConfig)
    print("Running finite-horizon DP...")
    V_timed = finite_horizon_dp(mdp, config)
    print("Done.")
    t = 0

    def pi(s: DiscreteState) -> DiscreteAction:
        """Assume that the policy is called once per time step."""
        nonlocal t
        V = V_timed[t]
        t += 1
        return value_function_to_greedy_policy(V, mdp, rng)(s)

    return pi
