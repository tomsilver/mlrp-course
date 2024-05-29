"""MDP Utilities."""

import abc
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import numpy as np

from mlrp_course.agent import Agent
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import Image


def bellman_backup(
    s: DiscreteState, V: Dict[DiscreteState, float], mdp: DiscreteMDP
) -> float:
    """Look ahead one step and propose an update for the value of s."""
    assert mdp.horizon is None
    vs = -float("inf")
    for a in mdp.action_space:
        qsa = 0.0
        for ns, p in mdp.get_transition_distribution(s, a).items():
            r = mdp.get_reward(s, a, ns)
            qsa += p * (r + mdp.temporal_discount_factor * V[ns])
        vs = max(qsa, vs)
    return vs


def value_to_action_value_function(
    V: Dict[DiscreteState, float], mdp: DiscreteMDP
) -> Dict[DiscreteState, Dict[DiscreteAction, float]]:
    """Convert a value (V) function to an action-value (Q) function."""
    Q: Dict[DiscreteState, Dict[DiscreteAction, float]] = {}

    S = mdp.state_space
    A = mdp.action_space
    gamma = mdp.temporal_discount_factor
    P = mdp.get_transition_probability
    R = mdp.get_reward

    for s in S:
        Q[s] = {}
        for a in A:
            Q[s][a] = sum(P(s, a, ns) * (R(s, a, ns) * gamma * V[ns]) for ns in S)

    return Q


def value_function_to_greedy_policy(
    V: Dict[DiscreteState, float], mdp: DiscreteMDP, rng: np.random.Generator
) -> Callable[[DiscreteState], DiscreteAction]:
    """Create a greedy policy given a value function."""
    gamma = mdp.temporal_discount_factor
    P = mdp.get_transition_distribution
    R = mdp.get_reward

    # Note: do not call value_to_action_value_function() here because we can
    # avoid enumerating the state space.
    def Q(s: DiscreteState, a: DiscreteAction) -> float:
        """Shorthand for the action-value function."""
        return sum(P(s, a)(ns) * (R(s, a, ns) + gamma * V[ns]) for ns in P(s, a))

    def pi(s: DiscreteState) -> DiscreteAction:
        """The greedy policy."""
        # Break ties randomly on actions.
        return max(mdp.action_space, key=lambda a: (Q(s, a), rng.uniform()))

    return pi


def sample_trajectory(
    initial_state: DiscreteState,
    policy: Callable[[DiscreteState], DiscreteAction],
    mdp: DiscreteMDP,
    max_horizon: int,
    rng: np.random.Generator,
) -> Tuple[List[DiscreteState], List[DiscreteAction]]:
    """Sample a trajectory by running a policy in an MDP."""
    states: List[DiscreteState] = [initial_state]
    actions: List[DiscreteAction] = []
    state = initial_state
    for _ in range(max_horizon):
        if mdp.state_is_terminal(state):
            break
        action = policy(state)
        state = mdp.sample_next_state(state, action, rng)
        actions.append(action)
        states.append(state)
    return states, actions


class DiscreteMDPGymEnv(gym.Env):
    """Convert an MDP into a gym environment to force RL-access only."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        mdp: DiscreteMDP,
        sample_initial_state: Callable[[int | None], DiscreteState],
    ) -> None:
        self._mdp = mdp
        self._sample_initial_state = sample_initial_state
        self._remaining_horizon = self._mdp.horizon or float("inf")
        self._current_state: DiscreteState | None = None  # set in reset()
        super().__init__()

    def reset(
        self, *, seed: int | None = None, options: Dict | None = None
    ) -> Tuple[DiscreteState, Dict]:
        super().reset(seed=seed)
        self._current_state = self._sample_initial_state(seed)
        self._remaining_horizon = self._mdp.horizon or float("inf")
        info: Dict = {}
        return self._current_state, info

    def step(
        self, action: DiscreteAction
    ) -> Tuple[DiscreteState, float, bool, bool, Dict]:
        next_state = self._mdp.sample_next_state(
            self._current_state, action, self.np_random
        )
        reward = self._mdp.get_reward(self._current_state, action, next_state)
        self._current_state = next_state
        terminated = self._mdp.state_is_terminal(self._current_state)
        self._remaining_horizon -= 1
        truncated = self._remaining_horizon <= 0
        info: Dict = {}
        return next_state, reward, terminated, truncated, info

    def render(self) -> Image:
        assert self.render_mode == "rgb_array"
        return self._mdp.render_state(self._current_state)


class DiscreteMDPAgent(Agent[DiscreteState, DiscreteAction], abc.ABC):
    """An agent acting in a DiscreteMDP."""

    def __init__(self, mdp: DiscreteMDP, seed: int) -> None:
        self._mdp = mdp
        super().__init__(seed)
