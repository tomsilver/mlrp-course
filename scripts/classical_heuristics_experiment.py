"""Compare various classical planning heuristics."""

from pathlib import Path
from typing import List, Tuple

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from mlrp_course.classical.algorithms.heuristics import (
    GoalCountHeuristic,
    HFFHeuristic,
    TrivialHeuristic,
)
from mlrp_course.classical.algorithms.search import SearchMetrics, run_astar, run_gbfs
from mlrp_course.classical.envs.pddl_problem import PDDLPlanningProblem
from mlrp_course.utils import load_pddl_asset

_HEURISTICS = {
    "hff": HFFHeuristic,
    "goal-count": GoalCountHeuristic,
    "trivial": TrivialHeuristic,
}

_SEARCH = {
    "gbfs": run_gbfs,
    "astar": run_astar,
}


def _run_single_trial(
    search_name: str,
    heuristic_name: str,
    domain_name: str,
    problem_idx: int,
) -> SearchMetrics:
    print("Running", search_name, heuristic_name, domain_name, problem_idx)
    # Create the problem.
    domain_str = load_pddl_asset(f"{domain_name}/domain.pddl")
    problem_str = load_pddl_asset(f"{domain_name}/problem{problem_idx}.pddl")
    problem = PDDLPlanningProblem.from_strings(domain_str, problem_str)
    # Create the search.
    # mypy has strange bug where it thinks heuristic is ABC
    heuristic = _HEURISTICS[heuristic_name](problem)  # type: ignore
    search = _SEARCH[search_name]
    # Run the search.
    _, _, metrics = search(problem, heuristic)
    return metrics


def _main(
    search_name: str,
    domain_name: str,
    num_problems: int,
    start_problem: int,
    outdir: Path,
    load: bool,
) -> None:
    csv_file = outdir / f"{domain_name}_{search_name}_heuristics_experiment.csv"
    if load:
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        return _df_to_plot(domain_name, search_name, df, outdir)
    columns = [
        "Search",
        "Heuristic",
        "Environment",
        "Problem",
        "Time",
        "Num Node Expansions",
        "Num Node Evals",
    ]
    results: List[Tuple[str, str, str, int, float, int, int]] = []
    for problem_idx in range(start_problem, start_problem + num_problems):
        for heuristic_name in _HEURISTICS:
            metrics = _run_single_trial(
                search_name, heuristic_name, domain_name, problem_idx
            )
            assert metrics.solved
            results.append(
                (
                    search_name,
                    heuristic_name,
                    domain_name,
                    problem_idx,
                    metrics.duration,
                    metrics.num_expansions,
                    metrics.num_evals,
                )
            )
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(csv_file)
    return _df_to_plot(domain_name, search_name, df, outdir)


def _df_to_plot(
    domain_name: str, search_name: str, df: pd.DataFrame, outdir: Path
) -> None:
    bar_order = list(_HEURISTICS)
    grouped = df.groupby("Heuristic")["Num Node Evals"].agg(["mean", "sem"])
    grouped = grouped.reindex(bar_order)
    matplotlib.rcParams.update({"font.size": 16})
    plt.figure()
    plt.bar(grouped.index, grouped["mean"], yerr=grouped["sem"], capsize=5)
    plt.xlabel("Heuristic")
    plt.ylabel("# Node Evals")
    plt.title(f"{domain_name}: {search_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    outfile = outdir / f"{domain_name}_{search_name}_heuristics_experiment.png"
    plt.savefig(outfile, dpi=350)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--search", default="gbfs", type=str)
    parser.add_argument("--domain", default="blocks", type=str)
    parser.add_argument("--num_problems", default=10, type=int)
    parser.add_argument("--start_problem", default=1, type=int)
    parser.add_argument("--outdir", default=Path("results"), type=Path)
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    _main(
        args.search,
        args.domain,
        args.num_problems,
        args.start_problem,
        args.outdir,
        args.load,
    )
