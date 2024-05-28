"""Tests for expectimax_search.py."""

import numpy as np

from mlrp_course.mdp.algorithms.expectimax_search import (
    ExpectimaxSearchConfig,
    expectimax_search,
)
from mlrp_course.mdp.envs.chase_mdp import ChaseState, ChaseWithRoomsMDP


def test_expectimax_search():
    """Tests for expectimax_search.py."""
    mdp = ChaseWithRoomsMDP()
    state = ChaseState((1, 1), ((3, 3),))
    rng = np.random.default_rng(123)
    config = ExpectimaxSearchConfig(search_horizon=10)
    for _ in range(10):
        if mdp.state_is_terminal(state):
            break
        action = expectimax_search(state, mdp, config)
        state = mdp.sample_next_state(state, action, rng)
    else:
        assert False, "Goal not reached"
