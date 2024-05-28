"""Solve finite-horizon MDPs by dynamic programming."""

from dataclasses import dataclass
from typing import Dict

from mlrp_course.agents import DiscreteMDPAgent
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import Hyperparameters
from mlrp_course.utils import value_function_to_greedy_policy


@dataclass(frozen=True)
class FiniteHorizonDPConfig(Hyperparameters):
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


class FiniteHorizonDPAgent(DiscreteMDPAgent):
    """An agent that plans offline with finite-horizon dynamic programming."""

    def __init__(self, planner_config: FiniteHorizonDPConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._value_fn = finite_horizon_dp(self._mdp, planner_config)

    def _get_action(self) -> DiscreteAction:
        assert self._last_observation is not None
        V = self._value_fn[self._timestep]
        pi = value_function_to_greedy_policy(V, self._mdp, self._rng)
        return pi(self._last_observation)
