"""Experiment with number of control points for GD in double integrator."""

import time
from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np
import pandas as pd
from jaxopt import GradientDescent

from mlrp_course.trajopt.algorithms.jaxopt_solver import (
    JaxOptTrajOptSolver,
    JaxOptTrajOptSolverHyperparameters,
)
from mlrp_course.trajopt.envs.double_integrator import (
    DoubleIntegratorHyperparameters,
    JaxDoubleIntegratorProblem,
)


def _main(
    start_seed: int,
    num_seeds: int,
    max_horizon: int,
    outdir: Path,
    load: bool,
) -> None:
    csv_file = outdir / "num_control_points_experiment.csv"
    if load:
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        return _df_to_plot(df, outdir)
    columns = ["Seed", "Num Control Points", "Cost", "Solve Time"]
    results: List[Tuple[int, int, float, float]] = []
    for seed in range(start_seed, start_seed + num_seeds):
        print(f"Starting {seed=}")
        for num_control_points in [2, 8, 32, 128, 512]:
            print(f"Starting {num_control_points=}")
            cost, duration = _run_single(seed, num_control_points, max_horizon)
            results.append((seed, num_control_points, cost, duration))
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(csv_file)
    return _df_to_plot(df, outdir)


def _run_single(
    seed: int,
    num_control_points: int,
    max_horizon: int,
) -> Tuple[float, float]:
    # pylint: disable=protected-access
    solver_config = JaxOptTrajOptSolverHyperparameters(
        num_control_points=num_control_points,
    )
    # Turn off JIT because otherwise it's trivially fast.
    solver_kwargs = {"jit": False, "maxiter": 100}
    solver = JaxOptTrajOptSolver(
        seed, GradientDescent, solver_kwargs, config=solver_config
    )
    env_config = DoubleIntegratorHyperparameters(horizon=max_horizon)
    solver_env = JaxDoubleIntegratorProblem(config=env_config)
    start_time = time.process_time()
    solver.reset(solver_env)
    traj = solver.solve()
    duration = time.process_time() - start_time
    cost = float(solver_env.get_traj_cost(traj))
    return cost, duration


def _df_to_plot(df: pd.DataFrame, outdir: Path) -> None:
    matplotlib.rcParams.update({"font.size": 20})
    fig_file = outdir / "num_control_points_experiment.png"

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_seed", default=0, type=int)
    parser.add_argument("--num_seeds", default=5, type=int)
    parser.add_argument("--max_horizon", default=25, type=int)
    parser.add_argument("--outdir", default=Path("results"), type=Path)
    parser.add_argument("--load", action="store_true")
    parser_args = parser.parse_args()
    _main(
        parser_args.start_seed,
        parser_args.num_seeds,
        parser_args.max_horizon,
        parser_args.outdir,
        parser_args.load,
    )
