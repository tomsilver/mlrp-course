"""Tests for sparse_sampling.py."""

import numpy as np

from mlrp_course.algorithms.sparse_sampling import sparse_sampling
from mlrp_course.mdp.chase_mdp import ChaseState, LargeChaseMDP


def test_sparse_sampling():
    """Tests for sparse_sampling.py."""
    mdp = LargeChaseMDP()
    state = ChaseState((1, 1), ((3, 3),))
    rng = np.random.default_rng(123)
    for _ in range(20):
        if mdp.state_is_terminal(state):
            break
        action = sparse_sampling(state, mdp, search_horizon=5, rng=rng)
        state = mdp.sample_next_state(state, action, rng)
    else:
        assert False, "Goal not reached"