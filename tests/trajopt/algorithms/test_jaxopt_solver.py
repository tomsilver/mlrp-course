"""Tests for jaxopt_solver.py."""

import numpy as np
from jaxopt import GradientDescent

from mlrp_course.trajopt.algorithms.jaxopt_solver import (
    JaxOptTrajOptSolverHyperparameters,
    JaxOptTrajOptSolver,
)
from mlrp_course.trajopt.algorithms.mpc_wrapper import MPCWrapper
from mlrp_course.trajopt.envs.double_integrator import (
    UnconstrainedDoubleIntegratorProblem,
)
from mlrp_course.trajopt.trajopt_problem import TrajOptTraj


def test_jaxopt_solver_trajopt():
    """Tests for jaxopt_solver.py."""
    config = JaxOptTrajOptSolverHyperparameters(
        num_control_points=3,
    )
    optimizer_cls = GradientDescent
    optimizer_kwargs = {"maxiter": 10}
    solver = JaxOptTrajOptSolver(
        optimizer_cls, optimizer_kwargs, seed=123, config=config
    )
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
