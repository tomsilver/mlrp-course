"""Tests for tiger_pomdp.py."""

import numpy as np

from mlrp_course.pomdp.envs.tiger_pomdp import TigerPOMDP


def test_tiger_pomdp():
    """Tests for tiger_pomdp.py."""
    pomdp = TigerPOMDP()
    assert "hear-left" in pomdp.observation_space
    assert "tiger-left" in pomdp.state_space
    assert "listen" in pomdp.action_space
    dist = pomdp.get_observation_distribution("listen", "tiger-left")
    assert np.isclose(sum(dist.values()), 1.0)
    assert np.isclose(dist["hear-left"], 0.85)
    dist = pomdp.get_observation_distribution("listen", "tiger-right")
    assert np.isclose(sum(dist.values()), 1.0)
    assert np.isclose(dist["hear-left"], 0.15)
    dist = pomdp.get_observation_distribution("open-left", "tiger-right")
    assert np.isclose(sum(dist.values()), 1.0)
    assert np.isclose(dist["none"], 1.0)
    assert not pomdp.state_is_terminal("tiger-left")
    dist = pomdp.get_transition_distribution("tiger-left", "listen")
    assert np.isclose(sum(dist.values()), 1.0)
    assert np.isclose(dist["tiger-left"], 1.0)
    reward = pomdp.get_reward("tiger-left", "listen", "tiger-left")
    assert np.isclose(reward, -1)
    reward = pomdp.get_reward("tiger-left", "open-left", "tiger-left")
    assert np.isclose(reward, -100)
    reward = pomdp.get_reward("tiger-left", "open-right", "tiger-left")
    assert np.isclose(reward, 10)
