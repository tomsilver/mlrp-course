"""Base classes for sequential decision-making agents."""

import abc
from typing import Generic, TypeVar

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
        self._timestep: int = 0
        self._train_or_eval = "eval"

    @abc.abstractmethod
    def _get_action(self) -> _ActType:
        """Produce an action to execute now."""

    def _learn_from_transition(
        self,
        obs: _ObsType,
        act: _ActType,
        next_obs: _ObsType,
        reward: float,
        done: bool,
    ) -> None:
        """Update any internal models based on the observed transition."""

    def reset(
        self,
        obs: _ObsType,
    ) -> _ActType:
        """Start a new episode."""
        self._last_observation = obs
        self._timestep = 0
        return self.step()

    def step(self) -> _ActType:
        """Get the next action to take."""
        self._last_action = self._get_action()
        self._timestep += 1
        return self._last_action

    def update(self, obs: _ObsType, reward: float, done: bool) -> None:
        """Record the reward and next observation following an action."""
        assert self._last_observation is not None
        assert self._last_action is not None
        self._learn_from_transition(
            self._last_observation, self._last_action, obs, reward, done
        )
        self._last_observation = obs

    def seed(self, seed: int) -> None:
        """Reset the random number generator."""
        self._rng = np.random.default_rng(seed)

    def train(self) -> None:
        """Switch to train mode."""
        self._train_or_eval = "train"

    def eval(self) -> None:
        """Switch to eval mode."""
        self._train_or_eval = "eval"


class DiscreteMDPAgent(Agent[DiscreteState, DiscreteAction], abc.ABC):
    """An agent acting in a DiscreteMDP."""

    def __init__(self, mdp: DiscreteMDP, *args, **kwargs) -> None:
        self._mdp = mdp
        super().__init__(*args, **kwargs)
