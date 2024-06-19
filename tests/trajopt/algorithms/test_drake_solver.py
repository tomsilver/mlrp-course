"""Tests for drake_solver.py."""

import numpy as np

from mlrp_course.trajopt.algorithms.drake_solver import (
    DrakeTrajOptSolver,
)
from mlrp_course.trajopt.algorithms.mpc_wrapper import MPCWrapper
from mlrp_course.trajopt.envs.double_integrator import (
    DrakeUnconstrainedDoubleIntegratorProblem,
    UnconstrainedDoubleIntegratorProblem,
)
from mlrp_course.trajopt.trajopt_problem import TrajOptTraj


def test_drake_solver_trajopt():
    """Tests for drake_solver.py."""
    seed = 123
    solver = DrakeTrajOptSolver(seed)
    mpc = MPCWrapper(solver)
    env = UnconstrainedDoubleIntegratorProblem()
    solver_env = DrakeUnconstrainedDoubleIntegratorProblem()
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
    # iio.mimsave("mpc_drake_double_integrator.mp4", imgs, fps=10)
