"""Monte Carlo Tree Search."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from mlrp_course.agents import DiscreteMDPAgent
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import AlgorithmConfig
from mlrp_course.utils import sample_trajectory


@dataclass(frozen=True)
class MCTSConfig(AlgorithmConfig):
    """Hyperparameters for MCTS."""

    search_horizon: int = 10
    num_simulations: int = 100
    explore_strategy: str = "ucb"
    max_rollout_length: int = 100
    num_rollouts: int = 3
    exploration_bonus: float = 5.0


def mcts(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    rng: np.random.Generator,
    config: MCTSConfig,
) -> DiscreteAction:
    """Monte Carlo Tree Search."""

    assert mdp.horizon is None, "MCTS only implemented for infinite-horizon"

    Q: Dict[DiscreteState, Dict[DiscreteAction, float]] = {}
    N: Dict[DiscreteState, Dict[DiscreteAction, int]] = {}

    for _ in range(config.num_simulations):
        _simulate(
            initial_state,
            mdp,
            config.search_horizon,
            Q,
            N,
            rng,
            config.explore_strategy,
            config.max_rollout_length,
            config.num_rollouts,
            config.exploration_bonus,
        )

    return max(mdp.action_space, key=lambda a: Q[initial_state].get(a, -float("inf")))


def _simulate(
    s: DiscreteState,
    mdp: DiscreteMDP,
    search_horizon: int,
    Q: Dict[DiscreteState, Dict[DiscreteAction, float]],
    N: Dict[DiscreteState, Dict[DiscreteAction, int]],
    rng: np.random.Generator,
    explore_strategy: str = "ucb",
    max_rollout_length: int = 100,
    num_rollouts: int = 3,
    exploration_bonus: float = 5.0,
) -> float:
    """Return an estimate of V(s) and update Q and N in-place."""
    A = sorted(mdp.action_space)  # sort for determinism
    gamma = mdp.temporal_discount_factor
    P = mdp.sample_next_state
    R = mdp.get_reward

    # If this state is terminal, return immediately.
    if search_horizon <= 0 or mdp.state_is_terminal(s):
        return 0.0

    # If this is the first time we're visiting this state, this is a leaf.
    if s not in N:
        Q[s] = {a: 0.0 for a in A}
        N[s] = {a: 0 for a in A}
        return _estimate_heuristic(s, mdp, rng, max_rollout_length, num_rollouts)

    # Get an action to try in this state, towards finding a leaf.
    a = _explore(s, mdp, Q, N, explore_strategy, exploration_bonus)
    ns = P(s, a, rng)

    # Recurse to find a leaf.
    q_sa = R(s, a, ns) + gamma * _simulate(
        ns,
        mdp,
        search_horizon - 1,
        Q,
        N,
        rng,
        explore_strategy,
        max_rollout_length,
        num_rollouts,
        exploration_bonus,
    )

    # Update the running averages.
    N[s][a] += 1
    Q[s][a] = ((N[s][a] - 1) * Q[s][a] + q_sa) / N[s][a]

    return Q[s][a]


def _estimate_heuristic(
    state: DiscreteState,
    mdp: DiscreteMDP,
    rng: np.random.Generator,
    max_rollout_length: int = 100,
    num_rollouts: int = 3,
) -> float:
    """Run rollouts with a random policy to get an estimate of V(s)."""

    A = sorted(mdp.action_space)  # sorting for determinism
    R = mdp.get_reward

    def pi(_: DiscreteState) -> DiscreteAction:
        """A random policy."""
        return A[rng.choice(len(A))]

    total_returns = 0.0
    for _ in range(num_rollouts):
        states, actions = sample_trajectory(state, pi, mdp, max_rollout_length, rng)
        for s, a, ns in zip(states[:-1], actions, states[1:], strict=True):
            total_returns += R(s, a, ns)
    return total_returns / num_rollouts


def _explore(
    s: DiscreteState,
    mdp: DiscreteMDP,
    Q: Dict[DiscreteState, Dict[DiscreteAction, float]],
    N: Dict[DiscreteState, Dict[DiscreteAction, int]],
    explore_strategy: str = "ucb",
    exploration_bonus: float = 5.0,
) -> DiscreteAction:
    assert explore_strategy == "ucb", "Explore strategy not implemented"
    n_s = sum(N[s].values())

    def _score_action(a: DiscreteAction) -> float:
        # Always take actions that have not yet been tried.
        if N[s][a] == 0:
            return float("inf")
        return Q[s][a] + exploration_bonus * np.sqrt(np.log(n_s) / N[s][a])

    # Break ties lexicographically.
    return max(mdp.action_space, key=lambda a: (_score_action(a), a))


class MCTSPAgent(DiscreteMDPAgent):
    """An agent that runs MCTS on every timestep."""

    def __init__(self, planner_config: MCTSConfig, *args, **kwargs) -> None:
        self._planner_config = planner_config
        super().__init__(*args, **kwargs)

    def _get_action(self) -> DiscreteAction:
        assert self._last_observation is not None
        return mcts(self._last_observation, self._mdp, self._rng, self._planner_config)
