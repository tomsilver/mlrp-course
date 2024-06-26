"""Tests for car_inspection_pomdp.py."""

import numpy as np

from mlrp_course.pomdp.envs.car_inspection_pomdp import CarInspectionPOMDP


def test_car_inspection_pomdp():
    """Tests for car_inspection_pomdp.py."""
    pomdp = CarInspectionPOMDP()
    assert "pass" in pomdp.observation_space
    assert "lemon" in pomdp.state_space
    assert "inspect" in pomdp.action_space
    dist = pomdp.get_observation_distribution("inspect", "lemon")
    assert np.isclose(dist["pass"], 0.4)
    dist = pomdp.get_observation_distribution("inspect", "peach")
    assert np.isclose(dist["pass"], 0.9)
    dist = pomdp.get_observation_distribution("buy", "lemon")
    assert np.isclose(dist["none"], 1.0)
    assert not pomdp.state_is_terminal("lemon")
    dist = pomdp.get_transition_distribution("lemon", "inspect")
    assert np.isclose(dist["lemon"], 1.0)
    reward = pomdp.get_reward("lemon", "buy", "lemon")
    assert np.isclose(reward, -100)
    reward = pomdp.get_reward("peach", "buy", "peach")
    assert np.isclose(reward, 60)
    reward = pomdp.get_reward("lemon", "inspect", "lemon")
    assert np.isclose(reward, -9)
    reward = pomdp.get_reward("peach", "dont-buy", "peach")
    assert np.isclose(reward, 0)
