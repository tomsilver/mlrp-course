"""Base classes for sequential decision-making agents."""

import abc
from typing import Dict, Generic, TypeVar

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.pomdp.discrete_pomdp import BeliefState, DiscreteObs, DiscretePOMDP
from mlrp_course.pomdp.utils import state_estimator

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
        if self._train_or_eval == "train":
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

    def __init__(self, mdp: DiscreteMDP, seed: int) -> None:
        self._mdp = mdp
        super().__init__(seed)


class DiscretePOMDPAgent(Agent[DiscreteObs, DiscreteAction], abc.ABC):
    """An agent acting in a DiscretePOMDP."""

    def __init__(self, pomdp: DiscretePOMDP, seed: int) -> None:
        self._pomdp = pomdp
        # Initialize with uniform belief.
        S = pomdp.state_space
        self._belief_state = BeliefState({s: 1.0 / len(S) for s in S})
        super().__init__(seed)

    def _get_initial_belief_state(self, obs: DiscreteObs) -> BeliefState:
        """Get a belief state based on an initial observation."""
        # P(s | o) = P(o |s) * P(s) / P(o).
        # Assume uniform P(s) and P(o).
        state_to_prob: Dict[DiscreteState, float] = {}
        for s in self._pomdp.state_space:
            dist = self._pomdp.get_initial_observation_distribution(s)
            state_to_prob[s] = dist[obs]
        z = sum(state_to_prob.values())
        assert z > 0, "Impossible initial observation"
        state_to_prob = {s: p / z for s, p in state_to_prob.items()}
        return BeliefState(state_to_prob)

    def _update_belief_state(self, obs: DiscreteObs) -> BeliefState:
        """Advance the belief state after receiving an observation."""
        assert self._last_action is not None
        return state_estimator(self._belief_state, self._last_action, obs, self._pomdp)

    def reset(
        self,
        obs: DiscreteObs,
    ) -> DiscreteAction:
        self._belief_state = self._get_initial_belief_state(obs)
        return super().reset(obs)

    def update(self, obs: DiscreteObs, reward: float, done: bool) -> None:
        self._belief_state = self._update_belief_state(obs)
        return super().update(obs, reward, done)
