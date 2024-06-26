"""Visualize the pendulum environment with a hand-coded policy."""

from pathlib import Path

import imageio.v2 as iio
import numpy as np
from tqdm import tqdm

from mlrp_course.trajopt.envs.pendulum import (
    PendulumHyperparameters,
    PendulumTrajOptProblem,
)
from mlrp_course.trajopt.trajopt_problem import TrajOptAction, TrajOptState, TrajOptTraj


def _policy(state: TrajOptState, env: PendulumTrajOptProblem) -> TrajOptAction:
    # Energy-shaping controller from https://underactuated.mit.edu/pend.html
    offset_theta, theta_dot = state
    theta = offset_theta - np.pi
    # pylint: disable=protected-access
    g = env._config.gravity
    m = env._config.mass
    l = env._config.length
    desired_energy = m * g * l
    current_energy = 0.5 * m * l**2 * theta_dot**2 - m * g * l * np.cos(theta)
    k = 100.0
    u = -k * theta_dot * (current_energy - desired_energy)
    u = np.clip(u, -2.0, 2.0)  # to make this nontrivial
    return np.array([u], dtype=np.float64)


def _main(max_horizon: int, outdir: Path, fps: int) -> None:
    env = PendulumTrajOptProblem(PendulumHyperparameters(horizon=max_horizon))
    initial_state = env.initial_state
    states = [initial_state]
    state = initial_state
    actions = []
    for _ in tqdm(range(env.horizon)):
        action = _policy(state, env)
        state = env.get_next_state(state, action)
        states.append(state)
        actions.append(action)
    traj = TrajOptTraj(np.array(states), np.array(actions))
    cost = env.get_traj_cost(traj)
    assert cost > 0

    imgs = [env.render_state(s) for s in states]
    outfile = outdir / "pendulum.mp4"
    iio.mimsave(outfile, imgs, fps=fps)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_horizon", default=200, type=int)
    parser.add_argument("--outdir", default=Path("results"), type=Path)
    parser.add_argument("--fps", default=20, type=int)
    args = parser.parse_args()
    _main(args.max_horizon, args.outdir, args.fps)
