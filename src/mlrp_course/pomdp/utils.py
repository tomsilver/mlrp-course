""""POMDP utilities."""

import abc
from collections import defaultdict
from typing import Dict, Optional, Set, TypeVar

from mlrp_course.agent import Agent
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.pomdp.discrete_pomdp import BeliefState, DiscreteObs, DiscretePOMDP
from mlrp_course.structs import CategoricalDistribution, Image


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
            # Keep the distribution sparse.
            if obs not in dist:
                continue
            state_to_prob[s] = dist(obs)
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
    ) -> None:
        self._belief_state = self._get_initial_belief_state(obs)
        return super().reset(obs)

    def update(self, obs: DiscreteObs, reward: float, done: bool) -> None:
        self._belief_state = self._update_belief_state(obs)
        return super().update(obs, reward, done)


class BeliefMDP(DiscreteMDP[BeliefState, DiscreteAction]):
    """A belief-space MDP induced by a POMDP."""

    def __init__(self, pomdp: DiscretePOMDP) -> None:
        # Remove terminal states from the POMDP. Otherwise, we could have bugs
        # where the agent repeatedly receives rewards from a terminal state
        # if there are other states in the belief state that are not terminal.
        self._pomdp = _POMDPWithoutTerminalStates(pomdp)
        super().__init__()

    @property
    def state_space(self) -> Set[BeliefState]:
        raise NotImplementedError("BeliefMDP state space cannot be enumerated")

    @property
    def action_space(self) -> Set[DiscreteAction]:
        return self._pomdp.action_space

    @property
    def temporal_discount_factor(self) -> float:
        return self._pomdp.temporal_discount_factor

    @property
    def horizon(self) -> Optional[int]:
        return self._pomdp.horizon

    def state_is_terminal(self, state: BeliefState) -> bool:
        return all(self._pomdp.state_is_terminal(s) for s in state)

    def get_reward(
        self, state: BeliefState, action: DiscreteAction, next_state: BeliefState
    ) -> float:
        P = self._pomdp.get_transition_distribution
        R = self._pomdp.get_reward
        return sum(
            p_st
            * sum(
                p_st1 * R(s_t, action, s_t1) for s_t1, p_st1 in P(s_t, action).items()
            )
            for s_t, p_st in state.items()
        )

    def get_transition_distribution(
        self, state: BeliefState, action: DiscreteAction
    ) -> CategoricalDistribution[BeliefState]:
        P = self._pomdp.get_transition_probability
        O = self._pomdp.get_observation_probability
        b_t = state
        a_t = action

        # Optimization: calculate all possible next states and observations.
        S_t1: Set[DiscreteState] = set()
        O_t1: Set[DiscreteObs] = set()
        for s_t in b_t:
            for s_t1 in self._pomdp.get_transition_distribution(s_t, a_t):
                S_t1.add(s_t1)
                O_t1.update(self._pomdp.get_observation_distribution(a_t, s_t1))

        dist: Dict[BeliefState, float] = defaultdict(float)

        for o_t1 in O_t1:
            b_t1 = state_estimator(b_t, a_t, o_t1, self._pomdp)
            # Pr(o_t1 | b_t, a_t).
            p = sum(
                O(a_t, s_t1, o_t1)
                * sum(P(s_t, a_t, s_t1) * p_st for s_t, p_st in b_t.items())
                for s_t1 in S_t1
            )
            dist[b_t1] = p

        return CategoricalDistribution(dist, normalize=True)

    def render_state(self, state: BeliefState) -> Image:
        raise NotImplementedError("Rendering not implemented for belief MDP")


def state_estimator(
    b_t: BeliefState,
    a_t: DiscreteAction,
    o_t1: DiscreteObs,
    pomdp: DiscretePOMDP,
) -> BeliefState:
    """State estimator for POMDPs."""
    P = pomdp.get_transition_probability
    O = pomdp.get_observation_probability

    # Optimization: calculate all possible next states.
    S_t1: Set[DiscreteState] = set()
    for s_t in b_t:
        S_t1.update(pomdp.get_transition_distribution(s_t, a_t))

    next_state_to_prob: Dict[DiscreteState, float] = {}
    for s_t1 in S_t1:
        next_state_to_prob[s_t1] = O(a_t, s_t1, o_t1) * sum(
            P(s_t, a_t, s_t1) * p_st1 for s_t, p_st1 in b_t.items()
        )

    return BeliefState(next_state_to_prob, normalize=True)


_O = TypeVar("_O", bound=DiscreteObs)
_S = TypeVar("_S", bound=DiscreteState)
_A = TypeVar("_A", bound=DiscreteAction)


class _POMDPWithoutTerminalStates(DiscretePOMDP[_O, _S, _A]):
    """A POMDP with terminal states effectively removed."""

    def __init__(self, pomdp: DiscretePOMDP[_O, _S, _A]) -> None:
        self._pomdp = pomdp

    @property
    def observation_space(self) -> Set[_O]:
        return self._pomdp.observation_space

    @property
    def state_space(self) -> Set[_S]:
        return self._pomdp.state_space

    @property
    def action_space(self) -> Set[_A]:
        return self._pomdp.action_space

    @property
    def temporal_discount_factor(self) -> float:
        return self._pomdp.temporal_discount_factor

    @property
    def horizon(self) -> Optional[int]:
        return self._pomdp.horizon

    def get_observation_distribution(
        self, action: _A, next_state: _S
    ) -> CategoricalDistribution[_O]:
        return self._pomdp.get_observation_distribution(action, next_state)

    def get_initial_observation_distribution(
        self, initial_state: _S
    ) -> CategoricalDistribution[_O]:
        return self._pomdp.get_initial_observation_distribution(initial_state)

    def state_is_terminal(self, state: _S) -> bool:
        # Never terminate.
        return False

    def get_reward(self, state: _S, action: _A, next_state: _S) -> float:
        # If we've terminated, don't give any more rewards.
        if self._pomdp.state_is_terminal(state):
            return 0.0
        return self._pomdp.get_reward(state, action, next_state)

    def get_transition_distribution(
        self, state: _S, action: _A
    ) -> CategoricalDistribution[_S]:
        # Extend the original transition distribution so that terminal states
        # always transition to themselves.
        if self._pomdp.state_is_terminal(state):
            return CategoricalDistribution({state: 1.0})
        return self._pomdp.get_transition_distribution(state, action)

    def render_state(self, state: _S) -> Image:
        return self._pomdp.render_state(state)
