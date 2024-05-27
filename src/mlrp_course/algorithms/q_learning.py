"""Q Learning."""

from dataclasses import dataclass
from typing import Collection, Dict

from mlrp_course.agents import Agent
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteState
from mlrp_course.structs import AlgorithmConfig


@dataclass(frozen=True)
class QLearningConfig(AlgorithmConfig):
    """Hyperparameters for Q Learning."""

    explore_strategy: str = "epsilon-greedy"
    epsilon: float = 0.1
    learning_rate: float = 0.1


class QLearningAgent(Agent):
    """An agent that learns with Q-learning."""

    def __init__(
        self,
        actions: Collection[DiscreteAction],
        gamma: float,
        config: QLearningConfig,
        *args,
        **kwargs,
    ) -> None:
        self._actions = sorted(actions)  # sorting for determinism
        self._gamma = gamma  # temporal discount factor from the env
        self._Q_dict: Dict[DiscreteState, Dict[DiscreteAction, float]] = {}
        self._config = config
        super().__init__(*args, **kwargs)

    def _Q(self, s: DiscreteState, a: DiscreteAction) -> float:
        if s not in self._Q_dict:
            # Randomly initialize to break ties.
            self._Q_dict[s] = {a: self._rng.uniform(-1e-3, 1e-3) for a in self._actions}
        return self._Q_dict[s][a]

    def _learn_from_transition(
        self,
        obs: DiscreteAction,
        act: DiscreteAction,
        next_obs: DiscreteAction,
        reward: float,
        done: bool,
    ) -> None:
        # TD learning.
        v_ns = 0.0 if done else max(self._Q(next_obs, a) for a in self._actions)
        target = reward + self._gamma * v_ns
        prediction = self._Q(obs, act)
        # Update in the direction of the target.
        alpha = self._config.learning_rate
        self._Q_dict[obs][act] = prediction + alpha * (target - prediction)
        return super()._learn_from_transition(obs, act, next_obs, reward, done)

    def _get_action(self) -> DiscreteAction:
        # Explore or exploit.
        if self._train_or_eval == "train":
            assert self._config.explore_strategy == "epsilon-greedy"
            if self._rng.uniform() < self._config.epsilon:
                # Choose a random action.
                return self._get_random_action()
        # Choose the best action.
        s = self._last_observation
        assert isinstance(s, DiscreteState)
        return max(self._actions, key=lambda a: self._Q(s, a))

    def _get_random_action(self) -> DiscreteAction:
        return self._actions[self._rng.choice(len(self._actions))]
