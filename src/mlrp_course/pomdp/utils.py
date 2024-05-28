""""POMDP utilities."""

from collections import defaultdict
from typing import Dict, Optional, Set

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.pomdp.discrete_pomdp import BeliefState, DiscreteObs, DiscretePOMDP
from mlrp_course.structs import Image


class BeliefMDP(DiscreteMDP[BeliefState, DiscreteAction]):
    """A belief-space MDP induced by a POMDP."""

    def __init__(self, pomdp: DiscretePOMDP) -> None:
        self._pomdp = pomdp
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
        return False

    def get_reward(
        self, state: BeliefState, action: DiscreteAction, next_state: BeliefState
    ) -> float:
        return get_belief_space_reward(state, action, next_state, self._pomdp)

    def get_transition_distribution(
        self, state: BeliefState, action: DiscreteAction
    ) -> Dict[BeliefState, float]:
        return get_belief_space_transition_distribution(state, action, self._pomdp)

    def render_state(self, state: BeliefState) -> Image:
        raise NotImplementedError("Rendering not implemented for belief MDP")


def get_belief_space_reward(
    b_t: BeliefState, a_t: DiscreteAction, b_t1: BeliefState, pomdp: DiscretePOMDP
) -> float:
    """Reward function in belief space for POMDPs."""
    R = pomdp.get_reward
    return sum(
        R(s_t, a_t, s_t1) * p_st * p_st1
        for s_t, p_st in b_t.items()
        for s_t1, p_st1 in b_t1.items()
    )


def get_belief_space_transition_distribution(
    b_t: BeliefState, a_t: DiscreteAction, pomdp: DiscretePOMDP
) -> Dict[BeliefState, float]:
    """Transition distribution in belief space for POMDPs."""
    P = pomdp.get_transition_probability
    O = pomdp.get_observation_probability

    # Optimization: calculate all possible next states.
    S_t1: Set[DiscreteState] = set()
    for s_t in b_t:
        S_t1.update(pomdp.get_transition_distribution(s_t, a_t))

    dist: Dict[BeliefState, float] = defaultdict(float)

    for o_t1 in pomdp.observation_space:
        b_t1 = state_estimator(b_t, a_t, o_t1, pomdp)
        # Pr(o_t1 | b_t, a_t).
        p = sum(
            O(s_t1, a_t, o_t1)
            * sum(P(s_t, a_t, s_t1) * p_st for s_t, p_st in b_t.items())
            for s_t1 in S_t1
        )
        dist[b_t1] = p

    return dist


def state_estimator(
    b_t: BeliefState, a_t: DiscreteAction, o_t1: DiscreteObs, pomdp: DiscretePOMDP
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
        next_state_to_prob[s_t1] = O(s_t1, a_t, o_t1) * sum(
            P(s_t, a_t, s_t1) * p_st1 for s_t, p_st1 in b_t.items()
        )

    # Normalize.
    z = sum(next_state_to_prob.values())
    next_state_to_prob = {s: p / z for s, p in next_state_to_prob.items()}

    return BeliefState(next_state_to_prob)
