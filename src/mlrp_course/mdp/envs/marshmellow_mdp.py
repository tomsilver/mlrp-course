"""The Marshmallow MDP described in lecture."""

from typing import Optional, Set, Tuple, TypeAlias

from mlrp_course.mdp.discrete_mdp import DiscreteMDP
from mlrp_course.structs import CategoricalDistribution, Image

MarshmellowState: TypeAlias = Tuple[int, bool]
MarshmellowAction: TypeAlias = str


class MarshmallowMDP(DiscreteMDP[MarshmellowState, MarshmellowAction]):
    """The Marshmallow MDP described in lecture."""

    @property
    def state_space(self) -> Set[MarshmellowState]:
        # (hunger level, marshmallow remains)
        return {(h, m) for h in (0, 1, 2) for m in (True, False)}

    @property
    def action_space(self) -> Set[MarshmellowAction]:
        return {"eat", "wait"}

    @property
    def horizon(self) -> Optional[int]:
        return 4

    def state_is_terminal(self, state: MarshmellowState) -> bool:
        # No terminal states here.
        return False

    def get_reward(
        self,
        state: MarshmellowState,
        action: MarshmellowAction,
        next_state: MarshmellowState,
    ) -> float:
        next_hunger_level = next_state[0]
        return -(next_hunger_level**2)

    def get_transition_distribution(
        self, state: MarshmellowState, action: MarshmellowAction
    ) -> CategoricalDistribution[MarshmellowState]:
        # Update marshmallow deterministically
        if action == "eat":
            next_m = False
        else:
            next_m = state[1]

        # Initialize next state distribution dict
        # Any state not included assumed to have 0 prob
        dist = {}
        for h in [0, 1, 2]:
            dist[(h, next_m)] = 0.0

        # Update hunger
        if action == "wait" or state[1] is False:
            # With 0.75 probability, hunger stays the same
            dist[(state[0], next_m)] += 0.75
            # With 0.25 probability, hunger increases by 1
            dist[(min(state[0] + 1, 2), next_m)] += 0.25

        else:
            assert action == "eat" and state[1] is True
            # Hunger deterministically set to 1 after eating
            dist[(0, next_m)] = 1.0

        return CategoricalDistribution(dist)

    def render_state(self, state: MarshmellowState) -> Image:
        raise NotImplementedError("Rendering not implemented for MDP.")
