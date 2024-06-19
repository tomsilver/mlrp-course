"""Tests for drake_solver.py."""

from mlrp_course.trajopt.algorithms.drake_solver import (
    DrakeTrajOptSolver,
)
from mlrp_course.trajopt.envs.pendulum import (
    DrakePendulumTrajOptProblem,
    PendulumHyperparameters,
    PendulumTrajOptProblem,
)


def test_drake_solver_trajopt():
    """Tests for drake_solver.py."""
    # NOTE: this is not stable; it fails for other seeds. Maybe add random
    # restarts in case of failure to drake optimizer in the future.
    seed = 123
    solver = DrakeTrajOptSolver(seed)
    env_config = PendulumHyperparameters(torque_lb=-2.0, torque_ub=2.0)
    env = PendulumTrajOptProblem(config=env_config)
    solver_env = DrakePendulumTrajOptProblem(config=env_config)
    solver.reset(solver_env)
    traj = solver.solve()
    cost = env.get_traj_cost(traj)
    assert cost > 0

    # Uncomment to visualize.
    # import imageio.v2 as iio
    # imgs = [env.render_state(s) for s in traj.states]
    # iio.mimsave("drake_pendulum.mp4", imgs, fps=10)
