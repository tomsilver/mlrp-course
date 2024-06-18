"""Tests for pendulum.py."""

import numpy as np

from mlrp_course.trajopt.envs.pendulum import UnconstrainedPendulumTrajOptProblem
from mlrp_course.trajopt.trajopt_problem import TrajOptTraj


def test_pendulum():
    """Tests for pendulum.py."""

    env = UnconstrainedPendulumTrajOptProblem(seed=123)
    initial_state = env.initial_state
    assert env.state_space.contains(initial_state)
    # Create a random trajectory.
    states = [initial_state]
    state = initial_state
    actions = []
    for _ in range(env.horizon):
        action = env.action_space.sample()
        state = env.get_next_state(state, action)
        states.append(state)
        actions.append(action)
    traj = TrajOptTraj(np.array(states), np.array(actions))
    cost = env.get_traj_cost(traj)
    assert cost > 0

    # Uncomment to visualize.
    # import imageio.v2 as iio
    # imgs = [env.render_state(s) for s in states]
    # iio.mimsave("pendulum.mp4", imgs, fps=10)
