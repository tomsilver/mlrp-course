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
    belief_state0 = BeliefState(
        {
            SearchAndRescueState((0, 1), (0, 0)): 0.5,
            SearchAndRescueState((0, 1), (0, 2)): 0.5,
        }
    )

    scan_left = SearchAndRescueAction("scan", (0, -1))
    assert scan_left in belief_mdp.action_space
    belief_state1 = BeliefState(
        {
            SearchAndRescueState((0, 1), (0, 0)): 0.9,
            SearchAndRescueState((0, 1), (0, 2)): 0.1,
        }
    )
    belief_state2 = BeliefState(
        {
            SearchAndRescueState((0, 1), (0, 0)): 0.1,
            SearchAndRescueState((0, 1), (0, 2)): 0.9,
        }
    )
    assert not belief_mdp.state_is_terminal(belief_state1)
    assert not belief_mdp.state_is_terminal(belief_state2)
    dist = belief_mdp.get_transition_distribution(belief_state0, scan_left)
    assert np.isclose(dist[belief_state1], 0.5)
    assert np.isclose(dist[belief_state2], 0.5)
    assert np.isclose(
        belief_mdp.get_reward(belief_state0, scan_left, belief_state1), -1.0
    )
    assert np.isclose(
        belief_mdp.get_reward(belief_state0, scan_left, belief_state2), -1.0
    )

    scan_right = SearchAndRescueAction("scan", (0, 1))
    assert scan_right in belief_mdp.action_space
    dist = belief_mdp.get_transition_distribution(belief_state0, scan_right)
    assert np.isclose(dist[belief_state1], 0.5)
    assert np.isclose(dist[belief_state2], 0.5)
    assert np.isclose(
        belief_mdp.get_reward(belief_state0, scan_right, belief_state1), -1.0
    )
    assert np.isclose(
        belief_mdp.get_reward(belief_state0, scan_right, belief_state2), -1.0
    )

    move_left = SearchAndRescueAction("move", (0, -1))
    assert move_left in belief_mdp.action_space
    belief_state3 = BeliefState(
        {
            SearchAndRescueState((0, 0), (0, 0)): 0.5,
            SearchAndRescueState((0, 0), (0, 2)): 0.5,
        }
    )
    belief_state4 = BeliefState(
        {
            SearchAndRescueState((0, 2), (0, 0)): 0.5,
            SearchAndRescueState((0, 2), (0, 2)): 0.5,
        }
    )
    assert not belief_mdp.state_is_terminal(belief_state3)
    assert not belief_mdp.state_is_terminal(belief_state4)
    dist = belief_mdp.get_transition_distribution(belief_state0, move_left)
    assert np.isclose(dist[belief_state3], 0.95)
    assert np.isclose(dist[belief_state4], 0.05)
    assert np.isclose(
        belief_mdp.get_reward(belief_state0, move_left, belief_state3), -1.0 + 0.5 * 100
    )
    assert np.isclose(
        belief_mdp.get_reward(belief_state0, move_left, belief_state4), -1.0 + 0.5 * 100
    )

    move_right = SearchAndRescueAction("move", (0, 1))
    assert move_right in belief_mdp.action_space
    dist = belief_mdp.get_transition_distribution(belief_state0, move_right)
    assert np.isclose(dist[belief_state3], 0.05)
    assert np.isclose(dist[belief_state4], 0.95)
    assert np.isclose(
        belief_mdp.get_reward(belief_state0, move_right, belief_state3),
        -1.0 + 0.5 * 100,
    )
    assert np.isclose(
        belief_mdp.get_reward(belief_state0, move_right, belief_state4),
        -1.0 + 0.5 * 100,
    )

    # Test tricky case where part of the belief state has terminated.
    dist = belief_mdp.get_transition_distribution(belief_state3, move_left)
    belief_state5 = BeliefState(
        {
            SearchAndRescueState((0, 1), (0, 2)): 1.0,
        }
    )
    belief_state6 = BeliefState(
        {
            SearchAndRescueState((0, 0), (0, 0)): 0.5 / (0.5 + 0.475),
            SearchAndRescueState((0, 0), (0, 2)): 0.475 / (0.5 + 0.475),
        }
    )
    assert np.isclose(dist[belief_state5], 0.05 * 0.5)
    assert np.isclose(dist[belief_state6], 1.0 - 0.05 * 0.5)

    # Test that the total rewards accrued over a long random rollout do not
    # ever exceed the theoretical maximum.
    returns = 0.0
    rng = np.random.default_rng(123)
    A = sorted(belief_mdp.action_space)
    b = belief_state0
    for _ in range(100):
        a = A[rng.choice(len(A))]
        nb = belief_mdp.sample_next_state(b, a, rng)
        r = belief_mdp.get_reward(b, a, nb)
        returns += r
        assert returns < 100.0
        b = nb
