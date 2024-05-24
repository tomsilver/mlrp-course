"""Tests for chase_mdp.py."""

import numpy as np

from mlrp_course.mdp.chase_mdp import ChaseMDP, ChaseState


def test_chase_mdp():
    """Tests for chase_mdp.py."""
    mdp = ChaseMDP()
    state0 = ChaseState((0, 0), ((1, 0),))
    state1 = ChaseState((0, 0), (None,))
    state2 = ChaseState((1, 0), (None,))
    state3 = ChaseState((1, 0), ((0, 0),))
    state4 = ChaseState((1, 0), ((1, 1),))
    assert state0 in mdp.state_space
    assert "up" in mdp.action_space
    assert not mdp.state_is_terminal(state0)
    assert mdp.state_is_terminal(state1)
    dist = mdp.get_transition_distribution(state0, "down")
    assert np.isclose(sum(dist.values()), 1.0)
    assert np.isclose(dist[state2], 0.75)
    assert np.isclose(dist[state3], 0.125)
    assert np.isclose(dist[state4], 0.125)
    img = mdp.render_state(state0)
    assert img.sum() > 0
