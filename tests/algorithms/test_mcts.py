"""Tests for mcts.py."""

import numpy as np

from mlrp_course.algorithms.mcts import mcts
from mlrp_course.mdp.chase_mdp import ChaseState, LargeChaseMDP


def test_mcts():
    """Tests for mcts.py."""
    mdp = LargeChaseMDP()
    state = ChaseState((1, 1), ((3, 3), (3, 4)))
    rng = np.random.default_rng(123)
    for _ in range(100):
        if mdp.state_is_terminal(state):
            break
        action = mcts(
            state,
            mdp,
            search_horizon=100,
            rng=rng,
            num_simulations=100,
            max_rollout_length=10,
        )
        state = mdp.sample_next_state(state, action, rng)
    else:
        assert False, "Goal not reached"
