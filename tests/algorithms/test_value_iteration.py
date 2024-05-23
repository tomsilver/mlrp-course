"""Tests for value_iteration.py."""

from mlrp_course.algorithms.value_iteration import value_iteration
from mlrp_course.mdp.chase_mdp import ChaseMDP


def test_value_iteration():
    """Tests for value_iteration.py."""
    mdp = ChaseMDP()
    Vs = value_iteration(mdp, max_num_iterations=100)
    assert len(Vs) < 100  # should be well less
    V = Vs[-1]
    assert V[((0, 0), (0, 1))] > V[((0, 0), (0, 2))]
    assert V[((1, 0), (1, 1))] > V[((1, 0), (1, 2))]
