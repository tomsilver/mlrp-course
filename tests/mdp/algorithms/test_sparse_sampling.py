"""Tests for sparse_sampling.py."""

import numpy as np

from mlrp_course.mdp.algorithms.sparse_sampling import (
    SparseSamplingHyperparameters,
    sparse_sampling,
)
from mlrp_course.mdp.envs.chase_mdp import ChaseState, LargeChaseMDP


def test_sparse_sampling():
    """Tests for sparse_sampling.py."""
    mdp = LargeChaseMDP()
    config = SparseSamplingHyperparameters(max_search_horizon=5)
    state = ChaseState((1, 1), ((3, 3),))
    rng = np.random.default_rng(123)
    for t in range(20):
        if mdp.state_is_terminal(state):
            break
        action = sparse_sampling(state, mdp, t, rng, config)
        state = mdp.sample_next_state(state, action, rng)
    else:
        assert False, "Goal not reached"
