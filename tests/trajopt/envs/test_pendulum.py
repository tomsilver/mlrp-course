"""Tests for pendulum.py."""

import numpy as np

from mlrp_course.trajopt.envs.pendulum import PendulumTrajOptProblem
from mlrp_course.trajopt.trajopt_problem import TrajOptTraj


def test_pendulum():
    """Tests for pendulum.py."""

    problem = PendulumTrajOptProblem(seed=123)
    initial_state = problem.initial_state
    assert problem.state_space.contains(initial_state)
    # Create a random trajectory.
    states = [initial_state]
    state = initial_state
    actions = []
    for _ in range(problem.horizon):
        action = problem.action_space.sample()
        state = problem.get_next_state(state, action)
        states.append(state)
        actions.append(action)
    traj = TrajOptTraj(np.array(states), np.array(actions))
    cost = problem.get_traj_cost(traj)
    assert cost > 0
