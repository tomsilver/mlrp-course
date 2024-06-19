"""Visualize gradient descent trajopt on the double integrator."""

from pathlib import Path
from typing import List, Tuple, Type

import numpy as np
from jaxopt import GradientDescent
from jaxopt._src.gradient_descent import ProxGradState
from matplotlib import animation
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from mlrp_course.trajopt.algorithms.jaxopt_solver import (
    JaxOptTrajOptSolver,
    JaxOptTrajOptSolverHyperparameters,
)
from mlrp_course.trajopt.envs.double_integrator import (
    DoubleIntegratorHyperparameters,
    DoubleIntegratorProblem,
)
from mlrp_course.utils import point_sequence_to_trajectory


def _main(
    seed: int,
    max_horizon: int,
    outdir: Path,
    fps: int,
    num_control_points: int,
) -> None:
    # pylint: disable=protected-access
    config = JaxOptTrajOptSolverHyperparameters(
        num_control_points=num_control_points,
    )
    dt = max_horizon / (config.num_control_points - 1)
    # Monkey patch to record all calls to the solver.
    solver_cls = GradientDescent

    history = []

    original_init_state = solver_cls.init_state

    def init_state(
        self: Type[GradientDescent], params: NDArray, *args, **kwargs
    ) -> ProxGradState:
        history.append((params, self.fun(params)))
        return original_init_state(self, params, *args, **kwargs)

    original_update = solver_cls.update

    def update(
        self: Type[GradientDescent], params: NDArray, *args, **kwargs
    ) -> Tuple[NDArray, ProxGradState]:
        history.append((params, self.fun(params)))
        return original_update(self, params, *args, **kwargs)

    solver_cls.init_state = init_state
    solver_cls.update = update

    # NOTE: we need to turn off jit for update to record as we want.
    solver_kwargs = {"verbose": True, "jit": False}

    solver = JaxOptTrajOptSolver(seed, solver_cls, solver_kwargs, config=config)
    env = DoubleIntegratorProblem(DoubleIntegratorHyperparameters(horizon=max_horizon))
    solver.reset(env)
    solver.solve()

    # Plot history.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax0, ax1 = axes.flat
    ts = np.arange(max_horizon + 1)
    params, loss = history[0]
    fig.suptitle(f"Iter = 0, Loss = {loss.item():.3f}")

    traj = point_sequence_to_trajectory(params, dt=dt)
    us = [traj(t) for t in ts]
    (control_line,) = ax0.plot(ts, us)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Control")
    ax0.set_ylim(-10, 10)

    states = [env.initial_state]
    for u in us[:-1]:
        states.append(env.get_next_state(states[-1], u))
    xs = np.array(states)[:, 0]
    (state_line,) = ax1.plot(ts, xs)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position")
    ax1.set_ylim(-2, 2)

    plt.tight_layout()

    def _update_plt(frame: int) -> List[plt.Line2D]:
        params, loss = history[frame]
        fig.suptitle(f"Iter = {frame}, Loss = {loss.item():.3f}")
        traj = point_sequence_to_trajectory(params, dt=dt)
        us = [traj(t) for t in ts]
        control_line.set_ydata(us)
        states = [env.initial_state]
        for u in us[:-1]:
            states.append(env.get_next_state(states[-1], u))
        xs = np.array(states)[:, 0]
        state_line.set_ydata(xs)
        return [control_line, state_line]

    frames = len(history)
    interval = 1000 / fps
    ani = animation.FuncAnimation(
        fig=fig, func=_update_plt, frames=frames, interval=interval
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
    parser.add_argument("--num_control_points", default=10, type=int)
    parser_args = parser.parse_args()
    _main(
        parser_args.seed,
        parser_args.max_horizon,
        parser_args.outdir,
        parser_args.fps,
        parser_args.num_control_points,
    )
