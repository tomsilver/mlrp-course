"""Tests for chase_mdp.py."""

import numpy as np

from mlrp_course.mdp.chase_mdp import ChaseMDP


def test_chase_mdp():
    """Tests for chase_mdp.py."""
    mdp = ChaseMDP()
    assert ((0, 0), (1, 0)) in mdp.state_space
    assert "up" in mdp.action_space
    assert not mdp.state_is_terminal(((0, 0), (1, 0)))
    assert mdp.state_is_terminal(((0, 0), (0, 0)))
    dist = mdp.get_transition_distribution(((0, 0), (1, 0)), "down")
    assert np.isclose(sum(dist.values()), 1.0)
    assert np.isclose(dist[((1, 0), (1, 0))], 0.75)
    assert np.isclose(dist[((1, 0), (0, 0))], 0.125)
    assert np.isclose(dist[((1, 0), (1, 1))], 0.125)
    img = mdp.render_state(((0, 0), (1, 0)))
    assert img.sum() > 0
