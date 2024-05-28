"""Car inspection POMDP from Leslie Kaelbling."""

from typing import Dict, Optional, Set, TypeAlias

from mlrp_course.pomdp.discrete_pomdp import DiscretePOMDP
from mlrp_course.structs import Image

CarInspectionObs: TypeAlias = str  # none, pass, fail
CarInspectionState: TypeAlias = str  # lemon or peach
CarInspectionAction: TypeAlias = str  # buy, wait, inspect


class CarInspectionPOMDP(
    DiscretePOMDP[CarInspectionObs, CarInspectionState, CarInspectionAction]
):
    """Car inspection POMDP from Leslie Kaelbling."""

    @property
    def observation_space(self) -> Set[CarInspectionObs]:
        return {"none", "pass", "fail"}

    @property
    def state_space(self) -> Set[CarInspectionState]:
        return {"lemon", "peach", "done"}

    @property
    def action_space(self) -> Set[CarInspectionAction]:
        return {"buy", "dont-buy", "inspect"}

    def get_observation_distribution(
        self, next_state: CarInspectionState, action: CarInspectionAction
    ) -> Dict[CarInspectionObs, float]:
        if action == "inspect":
            if next_state == "lemon":
                return {
                    "pass": 0.4,
                    "fail": 0.6,
                }
            assert next_state == "peach"
            return {
                "pass": 0.9,
                "fail": 0.1,
            }
        return {"none": 1.0}

    @property
    def horizon(self) -> Optional[int]:
        return 2

    def state_is_terminal(self, state: CarInspectionState) -> bool:
        # No terminal states here.
        return False

    def get_reward(
        self,
        state: CarInspectionState,
        action: CarInspectionAction,
        next_state: CarInspectionState,
    ) -> float:
        if action == "inspect":
            return -9  # $9 inspection fee
        if action == "dont-buy":
            return 0
        assert action == "buy"
        if state == "lemon":
            return -100
        assert state == "peach"
        return 60

    def get_transition_distribution(
        self, state: CarInspectionState, action: CarInspectionAction
    ) -> Dict[CarInspectionState, float]:
        # The state never changes in this very simple POMDP.
        return {state: 1.0}

    def render_state(self, state: CarInspectionState) -> Image:
        raise NotImplementedError("Rendering not implemented for POMDP.")
