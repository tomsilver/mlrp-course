"""The Zits MDP described in lecture."""

from typing import Dict, Set, TypeAlias

from mlrp_course.mdp.discrete_mdp import DiscreteMDP
from mlrp_course.structs import Image

ZitsState: TypeAlias = int
ZitsAction: TypeAlias = str


class ZitsMDP(DiscreteMDP[ZitsState, ZitsAction]):
    """The Zits MDP described in lecture."""

    @property
    def state_space(self) -> Set[ZitsState]:
        return {0, 1, 2, 3, 4}

    @property
    def action_space(self) -> Set[ZitsAction]:
        return {"apply", "sleep"}

    @property
    def temporal_discount_factor(self) -> float:
        return 0.9

    def state_is_terminal(self, state: ZitsState) -> bool:
        # No terminal states here.
        return False

    def get_reward(
        self, state: ZitsState, action: ZitsAction, next_state: ZitsState
    ) -> float:
        if action == "apply":
            return -1 - next_state
        assert action == "sleep"
        return -next_state

    def get_transition_distribution(
        self, state: ZitsState, action: ZitsAction
    ) -> Dict[ZitsState, float]:
        if action == "apply":
            return {0: 0.8, 4: 0.2}
        assert action == "sleep"
        return {min(state + 1, 4): 0.4, max(state - 1, 0): 0.6}

    def render_state(self, state: ZitsState) -> Image:
        raise NotImplementedError("Rendering not implemented for MDP.")
