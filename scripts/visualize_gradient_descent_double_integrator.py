"""Visualize gradient descent trajopt on the double integrator."""

from pathlib import Path

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from mlrp_course.trajopt.algorithms.gradient_descent import (
    GradientDescentHyperparameters,
    GradientDescentSolver,
)
from mlrp_course.trajopt.envs.double_integrator import DoubleIntegratorProblem
from mlrp_course.utils import point_sequence_to_trajectory


def _main(
    seed: int,
    max_horizon: int,
    outdir: Path,
    fps: int,
    num_control_points: int,
    learning_rate: float,
    num_descent_steps: int,
) -> None:
    # pylint: disable=protected-access
    config = GradientDescentHyperparameters(
        num_control_points=num_control_points,
        learning_rates=(learning_rate,),
        num_descent_steps=num_descent_steps,
    )
    dt = max_horizon / (config.num_control_points - 1)
    solver = GradientDescentSolver(seed, config=config)
    env = DoubleIntegratorProblem(seed=seed, horizon=max_horizon)
    solver.reset(env)
    solver.solve()
    history = solver._optimization_history

    # Plot history.
    fig, ax = plt.subplots(figsize=(5, 5))
    params, loss = history[0]
    traj = point_sequence_to_trajectory(params, dt=dt)
    xs = np.linspace(0, max_horizon, num=100, endpoint=True)
    ys = [traj(x) for x in xs]
    (line,) = ax.plot(xs, ys)
    ax.set_xlabel("Time")
    ax.set_ylabel("Control")
    ax.set_ylim(-1.1 * env._max_torque, 1.1 * env._max_torque)
    ax.set_title(f"Loss = {loss:.3f}")
    plt.tight_layout()

    def update(frame):
        params, loss = history[frame]
        traj = point_sequence_to_trajectory(params, dt=dt)
        ys = [traj(x) for x in xs]
        line.set_xdata(xs)
        line.set_ydata(ys)
        ax.set_title(f"Loss = {loss:.3f}")
        return (line,)

    frames = len(history)
    interval = 1000 / fps
    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval
    )
    outfile = outdir / f"gd_double_integrator_plot_seed{seed}.mp4"
    ani.save(outfile)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_horizon", default=25, type=int)
    parser.add_argument("--outdir", default=Path("results"), type=Path)
    parser.add_argument("--fps", default=20, type=int)
    parser.add_argument("--num_control_points", default=3, type=int)
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--num_descent_steps", default=100, type=int)
    args = parser.parse_args()
    _main(
        args.seed,
        args.max_horizon,
        args.outdir,
        args.fps,
        args.num_control_points,
        args.learning_rate,
        args.num_descent_steps,
    )
