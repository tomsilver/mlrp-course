"""Tests for POMDP utils."""

import numpy as np

from mlrp_course.pomdp.discrete_pomdp import BeliefState
from mlrp_course.pomdp.envs.search_and_rescue_pomdp import (
    SearchAndRescueAction,
    SearchAndRescueState,
    TinySearchAndRescuePOMDP,
)
from mlrp_course.pomdp.utils import BeliefMDP


def test_belief_mdp():
    """Tests for BeliefMDP."""
    pomdp = TinySearchAndRescuePOMDP()
    belief_mdp = BeliefMDP(pomdp)
    initial_belief_state = BeliefState(
        {
            SearchAndRescueState((0, 1), (0, 0)): 0.5,
            SearchAndRescueState((0, 1), (0, 2)): 0.5,
        }
    )

    scan_left = SearchAndRescueAction("scan", (0, -1))
    assert scan_left in belief_mdp.action_space
    next_belief_state1 = BeliefState(
        {
            SearchAndRescueState((0, 1), (0, 0)): 0.9,
            SearchAndRescueState((0, 1), (0, 2)): 0.1,
        }
    )
    next_belief_state2 = BeliefState(
        {
            SearchAndRescueState((0, 1), (0, 0)): 0.1,
            SearchAndRescueState((0, 1), (0, 2)): 0.9,
        }
    )
    dist = belief_mdp.get_transition_distribution(initial_belief_state, scan_left)
    assert np.isclose(dist[next_belief_state1], 0.5)
    assert np.isclose(dist[next_belief_state2], 0.5)

    scan_right = SearchAndRescueAction("scan", (0, 1))
    assert scan_right in belief_mdp.action_space
    dist = belief_mdp.get_transition_distribution(initial_belief_state, scan_right)
    assert np.isclose(dist[next_belief_state1], 0.5)
    assert np.isclose(dist[next_belief_state2], 0.5)

    move_left = SearchAndRescueAction("move", (0, -1))
    assert move_left in belief_mdp.action_space
    next_belief_state3 = BeliefState(
        {
            SearchAndRescueState((0, 0), (0, 0)): 0.5,
            SearchAndRescueState((0, 0), (0, 2)): 0.5,
        }
    )
    next_belief_state4 = BeliefState(
        {
            SearchAndRescueState((0, 2), (0, 0)): 0.5,
            SearchAndRescueState((0, 2), (0, 2)): 0.5,
        }
    )
    dist = belief_mdp.get_transition_distribution(initial_belief_state, move_left)
    assert np.isclose(dist[next_belief_state3], 0.95)
    assert np.isclose(dist[next_belief_state4], 0.05)

    move_right = SearchAndRescueAction("move", (0, 1))
    assert move_right in belief_mdp.action_space
    dist = belief_mdp.get_transition_distribution(initial_belief_state, move_right)
    assert np.isclose(dist[next_belief_state3], 0.05)
    assert np.isclose(dist[next_belief_state4], 0.95)
