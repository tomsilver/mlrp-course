"""Graph search algorithms for classical planning problems."""

from __future__ import annotations

import heapq as hq
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Tuple,
    TypeVar,
)

from mlrp_course.classical.envs.classical_problem import ClassicalPlanningProblem
from mlrp_course.structs import HashableComparable, Hyperparameters


@dataclass(frozen=True)
class HeuristicSearchHyperparameters(Hyperparameters):
    """Hyperparameters for heuristic search methods."""

    max_expansions: int = 10000000
    max_evals: int = 10000000
    timeout: float = 10000000


@dataclass(frozen=True)
class SearchMetrics:
    """Metrics about a search."""

    solved: bool
    duration: float
    num_expansions: int
    num_evals: int


_S = TypeVar("_S", bound=HashableComparable)
_A = TypeVar("_A", bound=HashableComparable)


@dataclass(frozen=True)
class _HeuristicSearchNode(Generic[_S, _A]):
    state: _S
    cumulative_cost: float
    parent: _HeuristicSearchNode[_S, _A] | None = None
    action: _A | None = None


class SearchFailure(Exception):
    """Raised when search fails to find a plan."""


def _run_heuristic_search(
    problem: ClassicalPlanningProblem[_S, _A],
    get_priority: Callable[[_HeuristicSearchNode[_S, _A]], Any],
    config: HeuristicSearchHyperparameters | None = None,
) -> Tuple[List[_S], List[_A], SearchMetrics]:
    """A generic heuristic search implementation.

    Depending on get_priority, can implement A*, GBFS, or UCS.

    If no goal is found, raises a SearchFailure exception.
    """
    if config is None:
        config = HeuristicSearchHyperparameters()

    queue: List[Tuple[Any, int, _HeuristicSearchNode[_S, _A]]] = []
    root_node: _HeuristicSearchNode[_S, _A] = _HeuristicSearchNode(
        problem.initial_state, 0
    )
    root_priority = get_priority(root_node)
    tiebreak = itertools.count()
    hq.heappush(queue, (root_priority, next(tiebreak), root_node))

    state_to_best_path_cost: Dict[_S, float] = defaultdict(lambda: float("inf"))
    state_to_best_path_cost[problem.initial_state] = 0

    num_expansions = 0
    num_evals = 1
    start_time = time.perf_counter()

    while (
        len(queue) > 0
        and time.perf_counter() - start_time < config.timeout
        and num_expansions < config.max_expansions
        and num_evals < config.max_evals
    ):
        _, _, node = hq.heappop(queue)
        # If we already found a better path here, don't bother.
        if state_to_best_path_cost[node.state] < node.cumulative_cost:
            continue
        # If the goal holds, return.
        if problem.check_goal(node.state):
            # Finished successfully!
            duration = time.perf_counter() - start_time
            metrics = SearchMetrics(
                solved=True,
                duration=duration,
                num_expansions=num_expansions,
                num_evals=num_evals,
            )
            states, actions = _finish_plan(node)
            return states, actions, metrics
        num_expansions += 1
        # Generate successors.
        for action, child_state, cost in problem.get_successors(node.state):
            if time.perf_counter() - start_time >= config.timeout:
                break
            child_path_cost = node.cumulative_cost + cost
            # If we already found a better path to this child, don't bother.
            if state_to_best_path_cost[child_state] <= child_path_cost:
                continue
            # Add new node.
            child_node = _HeuristicSearchNode(
                state=child_state,
                cumulative_cost=child_path_cost,
                parent=node,
                action=action,
            )
            priority = get_priority(child_node)
            num_evals += 1
            hq.heappush(queue, (priority, next(tiebreak), child_node))
            state_to_best_path_cost[child_state] = child_path_cost
            if num_evals >= config.max_evals:
                break

    raise SearchFailure("No plan found before timeout.")


def _finish_plan(node: _HeuristicSearchNode[_S, _A]) -> Tuple[List[_S], List[_A]]:
    """Helper for _run_heuristic_search."""
    rev_state_sequence: List[_S] = []
    rev_action_sequence: List[_A] = []

    while node.parent is not None:
        action = node.action
        assert action is not None
        rev_action_sequence.append(action)
        rev_state_sequence.append(node.state)
        node = node.parent
    rev_state_sequence.append(node.state)

    return rev_state_sequence[::-1], rev_action_sequence[::-1]


def run_gbfs(
    problem: ClassicalPlanningProblem[_S, _A],
    heuristic: Callable[[_S], float],
    config: HeuristicSearchHyperparameters | None = None,
) -> Tuple[List[_S], List[_A], SearchMetrics]:
    """Greedy best-first search."""
    get_priority = lambda n: heuristic(n.state)
    return _run_heuristic_search(problem, get_priority, config)


def run_astar(
    problem: ClassicalPlanningProblem[_S, _A],
    heuristic: Callable[[_S], float],
    config: HeuristicSearchHyperparameters | None = None,
) -> Tuple[List[_S], List[_A], SearchMetrics]:
    """A* search."""
    get_priority = lambda n: heuristic(n.state) + n.cumulative_cost
    return _run_heuristic_search(problem, get_priority, config)


def run_uniform_cost_search(
    problem: ClassicalPlanningProblem[_S, _A],
    config: HeuristicSearchHyperparameters | None = None,
) -> Tuple[List[_S], List[_A], SearchMetrics]:
    """Uniform-cost search."""
    get_priority = lambda n: n.cumulative_cost
    return _run_heuristic_search(problem, get_priority, config)
