"""Tests for policy_iteration.py."""

from mlrp_course.algorithms.policy_iteration import policy_iteration
from mlrp_course.mdp.chase_mdp import ChaseMDP, ChaseState


def test_policy_iteration():
    """Tests for policy_iteration.py."""
    mdp = ChaseMDP()
    Vs = policy_iteration(mdp, max_num_iterations=100)
    assert len(Vs) < 100  # should be well less
    V = Vs[-1]
    state0 = ChaseState((0, 0), ((0, 1),))
    state1 = ChaseState((0, 0), ((0, 2),))
    state2 = ChaseState((1, 0), ((1, 1),))
    state3 = ChaseState((1, 0), ((1, 2),))
    assert V[state0] > V[state1]
    assert V[state2] > V[state3]
