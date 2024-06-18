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
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax0, ax1 = axes.flat
    ts = np.arange(max_horizon + 1)
    params, loss = history[0]
    fig.suptitle(f"Iter = 0, Loss = {loss:.3f}")
    
    traj = point_sequence_to_trajectory(params, dt=dt)
    us = [np.clip(traj(t), -env._max_torque, env._max_torque) for t in ts]
    (control_line,) = ax0.plot(ts, us)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Control")
    ax0.set_ylim(-1.1 * env._max_torque, 1.1 * env._max_torque)
    
    states = [env.initial_state]
    for u in us[:-1]:
        states.append(env.get_next_state(states[-1], u))
    xs = np.array(states)[:, 0]
    (state_line,) = ax1.plot(ts, xs)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position")
    ax1.set_ylim(-2, 2)

    plt.tight_layout()

    def update(frame):
        params, loss = history[frame]
        fig.suptitle(f"Iter = {frame}, Loss = {loss:.3f}")
        traj = point_sequence_to_trajectory(params, dt=dt)
        us = [np.clip(traj(t), -env._max_torque, env._max_torque) for t in ts]
        control_line.set_ydata(us)
        states = [env.initial_state]
        for u in us[:-1]:
            states.append(env.get_next_state(states[-1], u))
        xs = np.array(states)[:, 0]
        state_line.set_ydata(xs)
        return (control_line, state_line)

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
    parser.add_argument("--learning_rate", default=1e-1, type=float)
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
