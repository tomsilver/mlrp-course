"""Run online planning with expectimax search."""

from dataclasses import dataclass
from functools import lru_cache

from mlrp_course.agents import DiscreteMDPAgent
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import AlgorithmConfig


@dataclass(frozen=True)
class ExpectimaxSearchConfig(AlgorithmConfig):
    """Hyperparameters for expectimax search."""

    search_horizon: int = 10


def expectimax_search(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    config: ExpectimaxSearchConfig,
) -> DiscreteAction:
    """Returns a single action to take."""
    # Note: no iteration over state space.
    A = mdp.action_space
    R = mdp.get_reward
    P = mdp.get_transition_distribution
    gamma = mdp.temporal_discount_factor

    @lru_cache(maxsize=None)
    def V(s, h):
        """Shorthand for the value function."""
        if h == config.search_horizon:
            return 0
        return max(Q(s, a, h) for a in A)

    @lru_cache(maxsize=None)
    def Q(s, a, h):
        """Shorthand for the action-value function."""
        return sum(P(s, a)[ns] * (R(s, a, ns) + gamma * V(ns, h + 1)) for ns in P(s, a))

    return max(A, key=lambda a: Q(initial_state, a, 0))


class ExpectimaxSearchAgent(DiscreteMDPAgent):
    """An agent that runs expectimax search at every timestep."""

    def __init__(self, planner_config: ExpectimaxSearchConfig, *args, **kwargs) -> None:
        self._planner_config = planner_config
        super().__init__(*args, **kwargs)

    def _get_action(self) -> DiscreteAction:
        assert self._last_observation is not None
        return expectimax_search(
            self._last_observation, self._mdp, self._planner_config
        )