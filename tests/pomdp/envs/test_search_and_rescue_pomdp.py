"""Tests for search_and_rescue_pomdp.py."""

import numpy as np

from mlrp_course.pomdp.envs.search_and_rescue_pomdp import (
    SearchAndRescueAction,
    SearchAndRescueObs,
    SearchAndRescuePOMDP,
    SearchAndRescueState,
)


def test_search_and_rescue_pomdp():
    """Tests for search_and_rescue_pomdp.py."""
    pomdp = SearchAndRescuePOMDP()
    assert SearchAndRescueObs((1, 2)) in pomdp.observation_space
    assert SearchAndRescueObs((1, 2), "got-response") in pomdp.observation_space
    assert SearchAndRescueState((1, 2), (0, 2)) in pomdp.state_space
    assert SearchAndRescueAction("move", (0, -1)) in pomdp.action_space
    dist = pomdp.get_observation_distribution(
        SearchAndRescueAction("scan", (-1, 0)),
        SearchAndRescueState((1, 2), (0, 2)),
    )
    assert np.isclose(dist[SearchAndRescueObs((1, 2), "got-response")], 0.9)
    dist = pomdp.get_observation_distribution(
        SearchAndRescueAction("scan", (0, -1)),
        SearchAndRescueState((1, 2), (0, 2)),
    )
    assert np.isclose(dist[SearchAndRescueObs((1, 2), "got-response")], 0.1)
    dist = pomdp.get_observation_distribution(
        SearchAndRescueAction("move", (-1, 0)),
        SearchAndRescueState((1, 2), (0, 2)),
    )
    assert np.isclose(dist[SearchAndRescueObs((1, 2))], 1.0)
    dist = pomdp.get_transition_distribution(
        SearchAndRescueState((1, 2), (0, 2)), SearchAndRescueAction("scan", (-1, 0))
    )
    assert np.isclose(dist[SearchAndRescueState((1, 2), (0, 2))], 1.0)
    dist = pomdp.get_transition_distribution(
        SearchAndRescueState((1, 2), (0, 2)), SearchAndRescueAction("move", (-1, 0))
    )
    assert np.isclose(dist[SearchAndRescueState((0, 2), (0, 2))], 0.9)
    assert np.isclose(dist[SearchAndRescueState((1, 2), (0, 2))], 0.1 / 3)
    assert np.isclose(dist[SearchAndRescueState((1, 1), (0, 2))], 0.1 / 3)
    assert np.isclose(dist[SearchAndRescueState((2, 2), (0, 2))], 0.1 / 3)
    assert pomdp.state_is_terminal(SearchAndRescueState((0, 2), (0, 2)))
    reward = pomdp.get_reward(
        SearchAndRescueState((1, 2), (0, 2)),
        SearchAndRescueAction("move", (-1, 0)),
        SearchAndRescueState((0, 2), (0, 2)),
    )
    assert np.isclose(reward, 100 - 1)
    reward = pomdp.get_reward(
        SearchAndRescueState((1, 2), (0, 2)),
        SearchAndRescueAction("move", (-1, 0)),
        SearchAndRescueState((1, 1), (0, 2)),
    )
    assert np.isclose(reward, -100 - 1)
    reward = pomdp.get_reward(
        SearchAndRescueState((1, 2), (0, 2)),
        SearchAndRescueAction("move", (-1, 0)),
        SearchAndRescueState((2, 2), (0, 2)),
    )
    assert np.isclose(reward, -1)
