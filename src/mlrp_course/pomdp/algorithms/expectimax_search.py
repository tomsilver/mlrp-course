"""Run online planning with expectimax search for POMDPs."""

from mlrp_course.mdp.algorithms.expectimax_search import (
    ExpectimaxSearchHyperparameters,
    expectimax_search,
)
from mlrp_course.mdp.discrete_mdp import DiscreteAction
from mlrp_course.pomdp.discrete_pomdp import BeliefState, DiscretePOMDP
from mlrp_course.pomdp.utils import BeliefMDP, DiscretePOMDPAgent


def pomdp_expectimax_search(
    initial_belief_state: BeliefState,
    pomdp: DiscretePOMDP,
    config: ExpectimaxSearchHyperparameters,
) -> DiscreteAction:
    """Returns a single action to take."""
    mdp = BeliefMDP(pomdp)
    return expectimax_search(initial_belief_state, mdp, config)


class POMDPExpectimaxSearchAgent(DiscretePOMDPAgent):
    """An agent that runs POMDP expectimax search at every timestep."""

    def __init__(
        self,
        pomdp: DiscretePOMDP,
        seed: int,
        expectimax_search_hyperparameters: (
            ExpectimaxSearchHyperparameters | None
        ) = None,
    ) -> None:
        self._expectimax_search_hyperparameters = (
            expectimax_search_hyperparameters or ExpectimaxSearchHyperparameters()
        )
        super().__init__(pomdp, seed)

    def _get_action(self) -> DiscreteAction:
        return pomdp_expectimax_search(
            self._belief_state, self._pomdp, self._expectimax_search_hyperparameters
        )
