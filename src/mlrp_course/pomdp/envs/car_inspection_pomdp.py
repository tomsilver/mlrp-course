"""Car inspection POMDP from Leslie Kaelbling."""

from dataclasses import dataclass
from typing import Dict, Optional, Set, TypeAlias

from mlrp_course.pomdp.discrete_pomdp import DiscretePOMDP
from mlrp_course.structs import Hyperparameters, Image

CarInspectionObs: TypeAlias = str  # none, pass, fail
CarInspectionState: TypeAlias = str  # lemon or peach
CarInspectionAction: TypeAlias = str  # buy, wait, inspect


@dataclass(frozen=True)
class CarInspectionPOMDPHyperparameters(Hyperparameters):
    """Hyperparameters for the CarInspectionPOMDP."""

    lemon_pass_prob: float = 0.4
    peach_pass_prob: float = 0.9
    inspection_fee: float = 9.0
    lemon_reward: float = -100.0
    peach_reward: float = 60.0


class CarInspectionPOMDP(
    DiscretePOMDP[CarInspectionObs, CarInspectionState, CarInspectionAction]
):
    """Car inspection POMDP from Leslie Kaelbling."""

    def __init__(self, config: CarInspectionPOMDPHyperparameters | None = None) -> None:
        self._config = config or CarInspectionPOMDPHyperparameters()

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
                    "pass": self._config.lemon_pass_prob,
                    "fail": 1 - self._config.lemon_pass_prob,
                }
            assert next_state == "peach"
            return {
                "pass": self._config.peach_pass_prob,
                "fail": 1 - self._config.peach_pass_prob,
            }
        return {"none": 1.0}

    @property
    def horizon(self) -> Optional[int]:
        return 3

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
            return -self._config.inspection_fee
        if action == "dont-buy":
            return 0
        assert action == "buy"
        if state == "lemon":
            return self._config.lemon_reward
        assert state == "peach"
        return self._config.peach_reward

    def get_transition_distribution(
        self, state: CarInspectionState, action: CarInspectionAction
    ) -> Dict[CarInspectionState, float]:
        # The state never changes in this very simple POMDP.
        return {state: 1.0}

    def render_state(self, state: CarInspectionState) -> Image:
        raise NotImplementedError("Rendering not implemented for POMDP.")
