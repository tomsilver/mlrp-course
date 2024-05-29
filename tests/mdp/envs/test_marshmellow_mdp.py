"""Tests for marshmellow_mdp.py."""

import numpy as np

from mlrp_course.mdp.envs.marshmellow_mdp import MarshmallowMDP


def test_marshmellow_mdp():
    """Tests for marshmellow_mdp.py."""
    mdp = MarshmallowMDP()
    assert (1, True) in mdp.state_space
    assert "eat" in mdp.action_space
    assert not mdp.state_is_terminal((1, True))
    dist = mdp.get_transition_distribution((1, True), "eat")
    assert np.isclose(dist[(0, False)], 1.0)
    dist = mdp.get_transition_distribution((1, False), "eat")
    assert np.isclose(dist[(1, False)], 0.75)
    assert np.isclose(dist[(2, False)], 0.25)
    reward = mdp.get_reward((1, True), "eat", (0, False))
    assert np.isclose(reward, 0.0)
    reward = mdp.get_reward((1, True), "wait", (1, True))
    assert np.isclose(reward, -1.0)
    reward = mdp.get_reward((1, True), "wait", (2, True))
    assert np.isclose(reward, -4.0)
