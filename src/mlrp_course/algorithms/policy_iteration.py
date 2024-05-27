"""Policy iteration."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from mlrp_course.agents import DiscreteMDPAgent
from mlrp_course.algorithms.policy_evaluation import evaluate_policy_linear_system
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import AlgorithmConfig
from mlrp_course.utils import (
    value_function_to_greedy_policy,
    value_to_action_value_function,
)


@dataclass(frozen=True)
class PolicyIterationConfig(AlgorithmConfig):
    """Hyperparameters for policy iteration."""

    max_num_iterations: int = 1000


def policy_iteration(
    mdp: DiscreteMDP,
    config: PolicyIterationConfig,
) -> List[Dict[DiscreteState, float]]:
    """Run policy iteration for a certain number of iterations or until there
    is no change to make.

    For analysis purposes, return the entire sequence of value function
    estimates.
    """
    # Sort for reproducibility.
    S = sorted(mdp.state_space)
    A = sorted(mdp.action_space)

    # Initialize a policy arbitrarily.
    arbitrary_action = A[0]
    pi = {s: arbitrary_action for s in S}

    all_estimates: List[Dict[DiscreteState, float]] = []

    for _ in range(config.max_num_iterations):
        # Compute the value function for the given policy.
        V = evaluate_policy_linear_system(pi.get, mdp)
        all_estimates.append(V)
        # Convert to action-value function.
        Q = value_to_action_value_function(V, mdp)  # type: ignore
        # Find a state and action that would yield a policy improvement.
        sa = _find_state_action_improvement(S, A, pi, Q)
        if sa is None:
            # Converged!
            break
        s, a = sa
        pi[s] = a

    return all_estimates


def _find_state_action_improvement(
    S: List[DiscreteState],
    A: List[DiscreteAction],
    pi: Dict[DiscreteState, DiscreteAction],
    Q: Dict[DiscreteState, Dict[DiscreteAction, float]],
) -> Optional[Tuple[DiscreteState, DiscreteAction]]:
    """Helper for policy iteration."""
    for s in S:
        pi_s = pi[s]
        for a in A:
            if a == pi_s:
                continue
            if Q[s][a] > Q[s][pi_s]:
                return (s, a)
    return None


class PolicyIterationAgent(DiscreteMDPAgent):
    """An agent that plans offline with policy iteration."""

    def __init__(self, planner_config: PolicyIterationConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        Vs = policy_iteration(self._mdp, planner_config)
        self._pi = value_function_to_greedy_policy(Vs[-1], self._mdp, self._rng)

    def _get_action(self) -> DiscreteAction:
        assert self._last_observation is not None
        return self._pi(self._last_observation)
