"""Tests for gradient_descent.py."""

import numpy as np

from mlrp_course.trajopt.algorithms.gradient_descent import (
    GradientDescentHyperparameters,
    GradientDescentSolver,
)
from mlrp_course.trajopt.algorithms.mpc_wrapper import MPCWrapper
from mlrp_course.trajopt.envs.double_integrator import (
    UnconstrainedDoubleIntegratorProblem,
)
from mlrp_course.trajopt.trajopt_problem import TrajOptTraj


def test_gradient_descent():
    """Tests for gradient_descent.py."""
    config = GradientDescentHyperparameters(
        num_control_points=3, learning_rates=[1e-3, 1e-2]
    )
    solver = GradientDescentSolver(123, config=config)
    mpc = MPCWrapper(solver)
    env = UnconstrainedDoubleIntegratorProblem(horizon=5)
    mpc.reset(env)
    # Run MPC to solve the problem.
    initial_state = env.initial_state
    states = [initial_state]
    state = initial_state
    actions = []
    for _ in range(env.horizon):
        action = mpc.step(state)
        state = env.get_next_state(state, action)
        states.append(state)
        actions.append(action)
    traj = TrajOptTraj(np.array(states), np.array(actions))
    cost = env.get_traj_cost(traj)
    assert cost > 0

    # Uncomment to visualize.
    # import imageio.v2 as iio
    # imgs = [env.render_state(s) for s in states]
    # iio.mimsave("mpc_gd_double_integrator.mp4", imgs, fps=10)
