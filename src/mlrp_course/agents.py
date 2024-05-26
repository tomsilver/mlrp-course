"""Base classes for sequential decision-making agents."""

import abc
from typing import Callable, Generic, TypeVar

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class Agent(Generic[_ObsType, _ActType]):
    """Base class for a sequential decision-making agent."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._last_observation: _ObsType | None = None
        self._last_action: _ActType | None = None

    @abc.abstractmethod
    def _get_action(self) -> _ActType:
        """Produce an action to execute now."""

    def _learn_from_transition(
        self, obs: _ObsType, act: _ActType, next_obs: _ObsType, reward: float
    ) -> None:
        """Update any internal models based on the observed transition."""

    def reset(
        self,
        obs: _ObsType,
    ) -> _ActType:
        """Start a new episode."""
        self._last_observation = obs
        return self.step()

    def step(self) -> _ActType:
        """Get the next action to take."""
        self._last_action = self._get_action()
        return self._last_action

    def update(self, obs: _ObsType, reward: float) -> None:
        """Record the reward and next observation following an action."""
        assert self._last_observation is not None
        assert self._last_action is not None
        self._learn_from_transition(
            self._last_observation, self._last_action, obs, reward
        )
        self._last_observation = obs

    def seed(self, seed: int) -> None:
        """Reset the random number generator."""
        self._rng = np.random.default_rng(seed)


class DiscreteMDPAgent(Agent[DiscreteState, DiscreteAction], abc.ABC):
    """An agent acting in a DiscreteMDP."""

    def __init__(self, mdp: DiscreteMDP, *args, **kwargs) -> None:
        self._mdp = mdp
        super().__init__(*args, **kwargs)


class OfflinePlanningDiscreteMDPAgent(DiscreteMDPAgent):
    """An agent that runs an offline planner to produce a policy."""

    def __init__(
        self,
        planning_alg: Callable[
            [DiscreteMDP], Callable[[DiscreteState], DiscreteAction]
        ],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pi = planning_alg(self._mdp)

    def _get_action(self) -> DiscreteAction:
        return self._pi(self._last_observation)


class OnlinePlanningDiscreteMDPAgent(DiscreteMDPAgent):
    """An agent that runs an online planner to produce a policy."""

    def __init__(
        self,
        planning_alg: Callable[
            [DiscreteState, DiscreteMDP], Callable[[DiscreteState], DiscreteAction]
        ],
        *args,
        **kwargs,
    ) -> None:
        self._planning_alg = planning_alg
        super().__init__(*args, **kwargs)
        self._pi: Callable[[DiscreteState], DiscreteAction] | None = None

    def _get_action(self) -> DiscreteAction:
        assert self._pi is not None
        return self._pi(self._last_observation)

    def reset(
        self,
        obs: DiscreteState,
    ) -> DiscreteAction:
        self._pi = self._planning_alg(obs, self._mdp)
        return super().reset(obs)
