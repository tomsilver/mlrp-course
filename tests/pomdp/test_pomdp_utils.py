"""Tests for POMDP utils."""

import numpy as np

from mlrp_course.pomdp.discrete_pomdp import BeliefState
from mlrp_course.pomdp.envs.search_and_rescue_pomdp import (
    SearchAndRescueAction,
    SearchAndRescuePOMDPHyperparameters,
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


def test_weirdness():
    """Temporary test to figure out weird behavior where belief MDP gets more
    reward than would ever be possible under POMDP."""
    pomdp = TinySearchAndRescuePOMDP(
        SearchAndRescuePOMDPHyperparameters(
            move_noise_probability=0.0, living_reward=-1, rescue_reward=100
        ),
    )
    belief_mdp = BeliefMDP(pomdp)
    belief_state = BeliefState(
        {
            SearchAndRescueState((0, 1), (0, 0)): 0.5,
            SearchAndRescueState((0, 1), (0, 2)): 0.5,
        }
    )
    move_left = SearchAndRescueAction("move", (0, -1))
    move_right = SearchAndRescueAction("move", (0, 1))
    rng = np.random.default_rng(124)
    print("Belief state 0:")
    print(belief_state)
    print("Action 0:", move_left)
    next_belief_state = belief_mdp.sample_next_state(belief_state, move_left, rng)
    print("Belief state 1:")
    print(next_belief_state)
    reward = belief_mdp.get_reward(belief_state, move_left, next_belief_state)
    print("Reward 0:", reward)
    # Total reward so far: 49
    assert np.isclose(reward, 0.5 * (100 - 1) + 0.5 * -1)
    belief_state = next_belief_state
    print("Action 1:", move_right)
    # NOTE: there are now two possible next outcomes. One where we observe
    # the robot moving to the right, and the other where the robot stays put.
    # Suppose we sample the former case (below).
    next_belief_state = BeliefState(
        {
            SearchAndRescueState((0, 1), (0, 2)): 1.0,
        }
    )
    next_belief_distribution = belief_mdp.get_transition_distribution(
        belief_state, move_right
    )
    assert np.isclose(next_belief_distribution[next_belief_state], 0.5)
    print("Belief state 2:", next_belief_state)
    reward = belief_mdp.get_reward(belief_state, move_right, next_belief_state)
    print("Reward 1:", reward)
    # Total reward so far: 48.5
    assert np.isclose(reward, 0.5 * (0) + 0.5 * -1)
    belief_state = next_belief_state
    print("Action 2:", move_right)
    next_belief_state = belief_mdp.sample_next_state(belief_state, move_right, rng)
    print("Belief state 3:", next_belief_state)
    reward = belief_mdp.get_reward(belief_state, move_right, next_belief_state)
    print("Reward 2:", reward)
    # Total reward so far: 147.5!!! This is surprising because the MAX reward
    # possible in the original POMDP is 99.
    assert np.isclose(reward, 1.0 * (100 - 1))
