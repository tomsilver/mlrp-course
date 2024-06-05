"""Compare various classical planning heuristics."""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

from mlrp_course.classical.algorithms.heuristics import TrivialHeuristic
from mlrp_course.classical.algorithms.search import SearchMetrics, run_astar, run_gbfs
from mlrp_course.classical.envs.pddl_problem import PDDLPlanningProblem
from mlrp_course.utils import load_pddl_asset

_HEURISTICS = {
    #   "goal-count": GoalCountHeuristic,
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
    # Create the problem.
    domain_str = load_pddl_asset(f"{domain_name}/domain.pddl")
    problem_str = load_pddl_asset(f"{domain_name}/problem{problem_idx}.pddl")
    problem = PDDLPlanningProblem(domain_str, problem_str)
    # Create the search.
    heuristic = _HEURISTICS[heuristic_name](problem)
    search = _SEARCH[search_name]
    # Run the search.
    _, _, metrics = search(problem, heuristic)
    import ipdb

    ipdb.set_trace()


def _main(
    search_name: str,
    domain_name: str,
    num_problems: int,
    start_problem: int,
    outdir: Path,
    load: bool,
) -> None:
    csv_file = outdir / f"{domain_name}_classical_heuristics_experiment.csv"
    if load:
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        return _df_to_plot(domain_name, df, outdir)
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
    return _df_to_plot(domain_name, df, outdir)


def _df_to_plot(domain_name: str, df: pd.DataFrame, outdir: Path) -> None:
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--search", default="gbfs", type=str)
    parser.add_argument("--domain", default="blocks", type=str)
    parser.add_argument("--num_problems", default=10, type=int)
    parser.add_argument("--start_problem", default=1, type=int)
    parser.add_argument("--outdir", default=Path("."), type=Path)
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
