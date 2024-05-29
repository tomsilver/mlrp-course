"""Tests for POMDP expectimax search."""

from mlrp_course.pomdp.algorithms.expectimax_search import (
    ExpectimaxSearchHyperparameters,
    pomdp_expectimax_search,
)
from mlrp_course.pomdp.discrete_pomdp import BeliefState
from mlrp_course.pomdp.envs.car_inspection_pomdp import (
    CarInspectionPOMDP,
    CarInspectionPOMDPHyperparameters,
)


def test_pomdp_expectimax_search():
    """Tests for POMDP expectimax search."""
    initial_belief = BeliefState({"lemon": 0.5, "peach": 0.5})
    search_config = ExpectimaxSearchHyperparameters(max_search_horizon=3)
    # By default, the cost of buying a lemon is really bad, and the inspection
    # cost is not so bad, so we should expect to inspect.
    pomdp = CarInspectionPOMDP()
    act = pomdp_expectimax_search(initial_belief, pomdp, 0, search_config)
    assert act == "inspect"
    # If the costs are different, we should expect to just buy.
    pomdp = CarInspectionPOMDP(
        CarInspectionPOMDPHyperparameters(
            inspection_fee=10000,
            lemon_reward=-10,
        )
    )
    act = pomdp_expectimax_search(initial_belief, pomdp, 0, search_config)
    assert act == "buy"
    # Now we expect to dont-buy.
    pomdp = CarInspectionPOMDP(
        CarInspectionPOMDPHyperparameters(
            inspection_fee=10000,
            lemon_reward=-100,
        )
    )
    act = pomdp_expectimax_search(initial_belief, pomdp, 0, search_config)
    assert act == "dont-buy"
    # Now we expect to dont-buy because inspection tells us nothing.
    pomdp = CarInspectionPOMDP(
        CarInspectionPOMDPHyperparameters(
            lemon_pass_prob=0.5,
            peach_pass_prob=0.5,
        )
    )
    act = pomdp_expectimax_search(initial_belief, pomdp, 0, search_config)
    assert act == "dont-buy"
    # If we're very confident that the car is a peach, we should buy.
    initial_belief = BeliefState({"lemon": 0.01, "peach": 0.99})
    pomdp = CarInspectionPOMDP()
    act = pomdp_expectimax_search(initial_belief, pomdp, 0, search_config)
    assert act == "buy"
