"""Q Learning."""

from dataclasses import dataclass
from typing import Collection, Dict

from mlrp_course.agents import DiscreteMDPAgent
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteState
from mlrp_course.structs import AlgorithmConfig


@dataclass(frozen=True)
class QLearningConfig(AlgorithmConfig):
    """Hyperparameters for Q Learning."""

    explore_strategy: str = "epsilon-greedy"
    epsilon: float = 0.1
    learning_rate: float = 0.1


class QLearningAgent(DiscreteMDPAgent):
    """An agent that learns with Q-learning."""

    def __init__(
        self,
        actions: Collection[DiscreteAction],
        gamma: float,
        planner_config: QLearningConfig,
        *args,
        **kwargs,
    ) -> None:
        self._actions = sorted(actions)  # sorting for determinism
        self._gamma = gamma  # temporal discount factor from the env
        self._Q: Dict[DiscreteState, Dict[DiscreteAction, float]] = {}
        self._planner_config = planner_config
        super().__init__(*args, **kwargs)

    def _learn_from_transition(
        self,
        obs: DiscreteAction,
        act: DiscreteAction,
        next_obs: DiscreteAction,
        reward: float,
    ) -> None:
        # TD learning.
        q_sa = self._Q[obs].get(act, 0.0)
        v_ns = max(self._Q[next_obs].values())
        alpha = self._planner_config.learning_rate
        self._Q[obs][act] = q_sa + alpha * (reward + self._gamma * v_ns - q_sa)
        return super()._learn_from_transition(obs, act, next_obs, reward)

    def _get_action(self) -> DiscreteAction:
        # TODO: explore vs. exploit
        assert self._planner_config.explore_strategy == "epsilon-greedy"
        if self._rng.uniform() < self._planner_config.epsilon:
            # Choose a random action.
            return self._get_random_action()
        # If this is the first time we're in the state, take a random action.
        assert self._last_observation is not None
        if self._last_observation not in self._Q:
            return self._get_random_action()
        # Choose the best action.
        q_s = self._Q[self._last_observation]
        candidates = set(q_s)
        return max(candidates, key=lambda a: q_s[a])

    def _get_random_action(self) -> DiscreteAction:
        return self._actions[self._rng.choice(len(self._actions))]
