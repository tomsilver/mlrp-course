"""Analyze number of samples vs predictive sampling performance."""

from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mlrp_course.trajopt.algorithms.mpc_wrapper import MPCWrapper
from mlrp_course.trajopt.algorithms.predictive_sampling import (
    PredictiveSamplingHyperparameters,
    PredictiveSamplingSolver,
)
from mlrp_course.trajopt.envs.pendulum import UnconstrainedPendulumTrajOptProblem
from mlrp_course.trajopt.trajopt_problem import TrajOptTraj


def _main(start_seed: int, num_seeds: int, outdir: Path, load: bool) -> None:
    csv_file = outdir / "predictive_sampling_experiments.csv"
    if load:
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        return _df_to_plot(df, outdir)
    columns = ["Seed", "Num Samples", "Cost"]
    results: List[Tuple[int, int, float]] = []
    for num_samples in [1, 10, 100]:
        print(f"Starting {num_samples=}")
        for seed in range(start_seed, start_seed + num_seeds):
            print(f"Starting {seed=}")
            result = _run_single(seed, num_samples)
            results.append((seed, num_samples, result))
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(csv_file)
    return _df_to_plot(df, outdir)


def _run_single(seed: int, num_samples: int) -> float:
    config = PredictiveSamplingHyperparameters(num_rollouts=num_samples)
    solver = PredictiveSamplingSolver(seed, config=config)
    mpc = MPCWrapper(solver)
    env = UnconstrainedPendulumTrajOptProblem(seed=seed)
    mpc.reset(env)
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
    return cost


def _df_to_plot(df: pd.DataFrame, outdir: Path) -> None:
    matplotlib.rcParams.update({"font.size": 20})
    fig_file = outdir / "predictive_sampling_experiments.png"

    grouped = df.groupby(["Num Samples"]).agg({"Cost": ["mean", "sem"]})
    grouped.columns = grouped.columns.droplevel(0)
    grouped = grouped.rename(columns={"mean": "Cost_mean", "sem": "Cost_sem"})
    grouped = grouped.reset_index()
    plt.figure(figsize=(10, 6))

    plt.plot(grouped["Num Samples"], grouped["Cost_mean"])
    plt.fill_between(
        grouped["Num Samples"],
        grouped["Cost_mean"] - grouped["Cost_sem"],
        grouped["Cost_mean"] + grouped["Cost_sem"],
        alpha=0.2,
    )

    plt.xlabel("Num Samples")
    plt.ylabel("Trajectory Cost")
    plt.title("Predictive Sampling (Pendulum)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150)
    print(f"Wrote out to {fig_file}")


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
