"""Compare various classical planning heuristics."""

from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict
from mlrp_course.classical.envs.pddl_problem import PDDLPlanningProblem
from mlrp_course.utils import load_pddl_asset


_HEURISTICS = ["goal-count", "trivial"]


def _run_single_trial(domain_name: str, problem_idx: int, heuristic_name: str) -> Dict[str, float]:
    # Create the problem.
    domain_str = load_pddl_asset(f"{domain_name}/domain.pddl")
    problem_str = load_pddl_asset(f"{domain_name}/problem{problem_idx}.pddl")
    problem = PDDLPlanningProblem(domain_str, problem_str)
    import ipdb; ipdb.set_trace()


def _main(domain_name: str, num_problems: int, start_problem: int, outdir: Path, load: bool) -> None:
    csv_file = outdir / f"{domain_name}_classical_heuristics_experiment.csv"
    if load:
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        return _df_to_plot(domain_name, df, outdir)
    columns = ["Environment", "Problem", "Heuristic", "Time", "Nodes Created"]
    results: List[Tuple[str, int, str, float, int]] = []
    for problem_idx in range(start_problem, start_problem + num_problems):
        for heuristic_name in _HEURISTICS:
            metrics = _run_single_trial(domain_name, problem_idx, heuristic_name)
            results.append((domain_name, problem_idx, heuristic_name, metrics["time"], metrics["nodes_created"]))
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(csv_file)
    return _df_to_plot(domain_name, df, outdir)

def _df_to_plot(domain_name: str, df: pd.DataFrame, outdir: Path) -> None:
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="blocks", type=str)
    parser.add_argument("--num_problems", default=10, type=int)
    parser.add_argument("--start_problem", default=1, type=int)
    parser.add_argument("--outdir", default=Path("."), type=Path)
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    _main(
        args.domain,
        args.num_problems,
        args.start_problem,
        args.outdir,
        args.load,
    )
