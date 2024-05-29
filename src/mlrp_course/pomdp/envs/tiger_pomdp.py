"""Tiger POMDP."""

from dataclasses import dataclass
from typing import Optional, Set, TypeAlias

from mlrp_course.pomdp.discrete_pomdp import DiscretePOMDP
from mlrp_course.structs import CategoricalDistribution, Hyperparameters, Image

TigerObs: TypeAlias = str  # none, hear-left, hear-right
TigerState: TypeAlias = str  # tiger-left or tiger-right
TigerAction: TypeAlias = str  # listen, open-left, open-right


@dataclass(frozen=True)
class TigerPOMDPHyperparameters(Hyperparameters):
    """Hyperparameters for the TigerPOMDP."""

    hearing_noise: float = 0.15  # probability of a misleading listen result
    listen_cost: float = 1.0
    bad_open_reward: float = -100.0
    good_open_reward: float = 10.0


class TigerPOMDP(DiscretePOMDP[TigerObs, TigerState, TigerAction]):
    """Tiger POMDP."""

    def __init__(self, config: TigerPOMDPHyperparameters | None = None) -> None:
        self._config = config or TigerPOMDPHyperparameters()

    @property
    def observation_space(self) -> Set[TigerObs]:
        return {"none", "hear-left", "hear-right"}

    @property
    def state_space(self) -> Set[TigerState]:
        return {"tiger-left", "tiger-right"}

    @property
    def action_space(self) -> Set[TigerAction]:
        return {"listen", "open-left", "open-right"}

    def get_observation_distribution(
        self,
        action: TigerAction,
        next_state: TigerState,
    ) -> CategoricalDistribution[TigerObs]:
        if action in ("open-left", "open-right"):
            return CategoricalDistribution({"none": 1.0})
        assert action == "listen"
        if next_state == "tiger-right":
            return CategoricalDistribution(
                {
                    "hear-left": self._config.hearing_noise,
                    "hear-right": 1.0 - self._config.hearing_noise,
                }
            )
        assert next_state == "tiger-left"
        return CategoricalDistribution(
            {
                "hear-left": 1.0 - self._config.hearing_noise,
                "hear-right": self._config.hearing_noise,
            }
        )

    @property
    def horizon(self) -> Optional[int]:
        return 3

    def state_is_terminal(self, state: TigerState) -> bool:
        # No terminal states here.
        return False

    def get_reward(
        self,
        state: TigerState,
        action: TigerAction,
        next_state: TigerState,
    ) -> float:
        if action == "listen":
            return -self._config.listen_cost
        assert action.startswith("open-")
        assert state.startswith("tiger-")
        action_direction = action[len("open-") :]
        state_direction = state[len("tiger-") :]
        assert action_direction in ("left", "right")
        assert state_direction in ("left", "right")
        if action_direction == state_direction:
            return self._config.bad_open_reward
        return self._config.good_open_reward

    def get_transition_distribution(
        self, state: TigerState, action: TigerAction
    ) -> CategoricalDistribution[TigerState]:
        # The state never changes in this very simple POMDP.
        return CategoricalDistribution({state: 1.0})

    def render_state(self, state: TigerState) -> Image:
        raise NotImplementedError("Rendering not implemented for POMDP.")
