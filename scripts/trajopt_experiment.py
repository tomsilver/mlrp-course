"""Compare different trajopt solvers."""

from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from mlrp_course.trajopt.algorithms.jaxopt_solver import (
    JaxOptTrajOptSolver,
)
from mlrp_course.trajopt.algorithms.mpc_wrapper import MPCWrapper
from mlrp_course.trajopt.algorithms.predictive_sampling import (
    PredictiveSamplingSolver,
)
from mlrp_course.trajopt.envs.pendulum import UnconstrainedPendulumTrajOptProblem
from mlrp_course.trajopt.trajopt_problem import TrajOptTraj

_SOLVERS = {
    "Gradient Descent": JaxOptTrajOptSolver,
    "Predictive Sampling": PredictiveSamplingSolver,
}


def _main(start_seed: int, num_seeds: int, outdir: Path, load: bool) -> None:
    csv_file = outdir / "trajopt_experiment.csv"
    if load:
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        return _df_to_plot(df, outdir)
    columns = ["Seed", "Solver", "Cost"]
    results: List[Tuple[int, str, float]] = []
    for seed in range(start_seed, start_seed + num_seeds):
        print(f"Starting {seed=}")
        for solver in _SOLVERS:
            print(f"Starting {solver=}")
            result = _run_single(seed, solver)
            results.append((seed, solver, result))
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(csv_file)
    return _df_to_plot(df, outdir)


def _run_single(seed: int, solver_name: str) -> float:
    solver = _SOLVERS[solver_name](seed)
    mpc = MPCWrapper(solver)
    env = UnconstrainedPendulumTrajOptProblem()
    mpc.reset(env)
    initial_state = env.initial_state
    states = [initial_state]
    state = initial_state
    actions = []
    for _ in tqdm(range(env.horizon)):
        action = mpc.step(state)
        state = env.get_next_state(state, action)
        states.append(state)
        actions.append(action)
    traj = TrajOptTraj(np.array(states), np.array(actions))
    cost = float(env.get_traj_cost(traj))
    return cost


def _df_to_plot(df: pd.DataFrame, outdir: Path) -> None:
    matplotlib.rcParams.update({"font.size": 20})
    outfile = outdir / "trajopt_experiment.png"

    bar_order = list(_SOLVERS)
    grouped = df.groupby("Solver")["Cost"].agg(["mean", "sem"])
    grouped = grouped.reindex(bar_order)
    matplotlib.rcParams.update({"font.size": 16})
    plt.figure()
    plt.bar(grouped.index, grouped["mean"], yerr=grouped["sem"], capsize=5)
    plt.xlabel("Solver")
    plt.ylabel("Cost")
    plt.title("Pendulum TrajOpt")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outfile, dpi=350)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_seeds", default=5, type=int)
    parser.add_argument("--outdir", default=Path("results"), type=Path)
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    _main(
        args.seed,
        args.num_seeds,
        args.outdir,
        args.load,
    )
