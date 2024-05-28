"""Tests for zits_mdp.py."""

import numpy as np

from mlrp_course.mdp.envs.zits_mdp import ZitsMDP


def test_zits_mdp():
    """Tests for zits_mdp.py."""
    mdp = ZitsMDP()
    assert 1 in mdp.state_space
    assert "apply" in mdp.action_space
    assert not mdp.state_is_terminal(1)
    dist = mdp.get_transition_distribution(2, "apply")
    assert np.isclose(sum(dist.values()), 1.0)
    assert np.isclose(dist[0], 0.8)
    assert np.isclose(dist[4], 0.2)
    dist = mdp.get_transition_distribution(2, "sleep")
    assert np.isclose(sum(dist.values()), 1.0)
    assert np.isclose(dist[1], 0.6)
    assert np.isclose(dist[3], 0.4)
    reward = mdp.get_reward(2, "sleep", 1)
    assert np.isclose(reward, -1)
    reward = mdp.get_reward(2, "sleep", 3)
    assert np.isclose(reward, -3)
    reward = mdp.get_reward(2, "apply", 4)
    assert np.isclose(reward, -5)
