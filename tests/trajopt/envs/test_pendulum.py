"""Tests for pendulum.py."""

from mlrp_course.trajopt.envs.pendulum import PendulumTrajOptProblem


def test_pendulum():
    """Tests for pendulum.py."""

    problem = PendulumTrajOptProblem()
    initial_state = problem.initial_state
    assert problem.state_space.contains(initial_state)
