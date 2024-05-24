"""Tests for policy_iteration.py."""

from mlrp_course.algorithms.policy_iteration import policy_iteration
from mlrp_course.mdp.chase_mdp import ChaseMDP


def test_policy_iteration():
    """Tests for policy_iteration.py."""
    mdp = ChaseMDP()
    Vs = policy_iteration(mdp, max_num_iterations=100)
    assert len(Vs) < 100  # should be well less
    V = Vs[-1]
    assert V[((0, 0), (0, 1))] > V[((0, 0), (0, 2))]
    assert V[((1, 0), (1, 1))] > V[((1, 0), (1, 2))]
