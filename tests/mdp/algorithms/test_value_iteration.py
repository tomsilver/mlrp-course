"""Tests for value_iteration.py."""

from mlrp_course.mdp.algorithms.value_iteration import (
    ValueIterationHyperparameters,
    value_iteration,
)
from mlrp_course.mdp.envs.chase_mdp import ChaseMDP, ChaseState


def test_value_iteration():
    """Tests for value_iteration.py."""
    mdp = ChaseMDP()
    config = ValueIterationHyperparameters(max_num_iterations=100)
    Vs = value_iteration(mdp, config)
    assert len(Vs) < 100  # should be well less
    V = Vs[-1]
    state0 = ChaseState((0, 0), ((0, 1),))
    state1 = ChaseState((0, 0), ((0, 2),))
    state2 = ChaseState((1, 0), ((1, 1),))
    state3 = ChaseState((1, 0), ((1, 2),))
    assert V[state0] > V[state1]
    assert V[state2] > V[state3]
