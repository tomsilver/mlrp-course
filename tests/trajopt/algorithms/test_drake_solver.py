"""Tests for drake_solver.py."""

from mlrp_course.trajopt.algorithms.drake_solver import (
    DrakeTrajOptSolver,
)
from mlrp_course.trajopt.envs.double_integrator import (
    DoubleIntegratorHyperparameters,
    DoubleIntegratorProblem,
    DrakeDoubleIntegratorProblem,
)


def test_drake_solver_trajopt():
    """Tests for drake_solver.py."""
    seed = 123
    solver = DrakeTrajOptSolver(seed)
    env_config = DoubleIntegratorHyperparameters(torque_lb=-1.0, torque_ub=1.0)
    env = DoubleIntegratorProblem(config=env_config)
    solver_env = DrakeDoubleIntegratorProblem(config=env_config)
    solver.reset(solver_env)
    traj = solver.solve()
    cost = env.get_traj_cost(traj)
    assert cost > 0

    # Uncomment to visualize.
    # import imageio.v2 as iio
    # imgs = [env.render_state(s) for s in traj.states]
    # iio.mimsave("drake_double_integrator.mp4", imgs, fps=10)
