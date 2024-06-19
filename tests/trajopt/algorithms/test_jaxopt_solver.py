"""Tests for jaxopt_solver.py."""

import numpy as np
from jaxopt import GradientDescent

from mlrp_course.trajopt.algorithms.jaxopt_solver import (
    JaxOptTrajOptSolver,
    JaxOptTrajOptSolverHyperparameters,
)
from mlrp_course.trajopt.algorithms.mpc_wrapper import MPCWrapper
from mlrp_course.trajopt.envs.double_integrator import (
    DoubleIntegratorProblem,
    JaxDoubleIntegratorProblem,
)
from mlrp_course.trajopt.trajopt_problem import TrajOptTraj


def test_jaxopt_solver_trajopt():
    """Tests for jaxopt_solver.py."""
    config = JaxOptTrajOptSolverHyperparameters(
        num_control_points=3,
    )
    optimizer_cls = GradientDescent
    optimizer_kwargs = {"maxiter": 10}
    seed = 123
    horizon = 5
    solver = JaxOptTrajOptSolver(seed, optimizer_cls, optimizer_kwargs, config=config)
    mpc = MPCWrapper(solver)
    env = DoubleIntegratorProblem(horizon=horizon)
    solver_env = JaxDoubleIntegratorProblem(horizon=horizon)
    mpc.reset(solver_env)
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
