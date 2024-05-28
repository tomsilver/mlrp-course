"""Run online planning with expectimax search for POMDPs."""

from mlrp_course.mdp.algorithms.expectimax_search import (
    ExpectimaxSearchConfig,
    expectimax_search,
)
from mlrp_course.mdp.discrete_mdp import DiscreteAction
from mlrp_course.pomdp.discrete_pomdp import BeliefState, DiscretePOMDP
from mlrp_course.pomdp.utils import BeliefMDP


def pomdp_expectimax_search(
    initial_belief_state: BeliefState,
    pomdp: DiscretePOMDP,
    config: ExpectimaxSearchConfig,
) -> DiscreteAction:
    """Returns a single action to take."""
    mdp = BeliefMDP(pomdp)
    return expectimax_search(initial_belief_state, mdp, config)
