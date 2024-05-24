"""Tests for expectimax_search.py."""

import numpy as np

from mlrp_course.algorithms.expectimax_search import expectimax_search
from mlrp_course.mdp.chase_mdp import ChaseWithRoomsMDP


def test_expectimax_search():
    """Tests for expectimax_search.py."""
    mdp = ChaseWithRoomsMDP()
    state = ((1, 1), (3, 3))
    rng = np.random.default_rng(123)
    for _ in range(10):
        if mdp.state_is_terminal(state):
            break
        action = expectimax_search(state, mdp, search_horizon=10)
        state = mdp.sample_next_state(state, action, rng)
    else:
        assert False, "Goal not reached"
