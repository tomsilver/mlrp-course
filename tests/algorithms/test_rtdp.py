"""Tests for rtdp.py."""

import numpy as np

from mlrp_course.algorithms.rtdp import rtdp
from mlrp_course.mdp.chase_mdp import ChaseWithRoomsMDP


def test_rtdp():
    """Tests for rtdp.py."""
    mdp = ChaseWithRoomsMDP()
    state = ((1, 1), (3, 3))
    rng = np.random.default_rng(123)
    search_horizon = 10
    for _ in range(10):
        if mdp.state_is_terminal(state):
            break
        action = rtdp(state, mdp, search_horizon, rng)
        state = mdp.sample_next_state(state, action, rng)
    else:
        assert False, "Goal not reached"
