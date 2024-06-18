"""Tests for predictive_sampling.py."""

import numpy as np

from mlrp_course.trajopt.algorithms.mpc_wrapper import MPCWrapper
from mlrp_course.trajopt.algorithms.predictive_sampling import (
    PredictiveSamplingHyperparameters,
    PredictiveSamplingSolver,
)
from mlrp_course.trajopt.envs.pendulum import UnconstrainedPendulumTrajOptProblem
from mlrp_course.trajopt.trajopt_problem import TrajOptTraj


def test_predictive_sampling():
    """Tests for predictive_sampling.py."""
    # Use small number of rollouts for faster unit test.
    config = PredictiveSamplingHyperparameters(num_rollouts=5, num_control_points=3)
    solver = PredictiveSamplingSolver(123, config=config)
    mpc = MPCWrapper(solver)
    env = UnconstrainedPendulumTrajOptProblem(seed=123, horizon=10)
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
    # iio.mimsave("mpc_ps_pendulum.mp4", imgs, fps=10)
