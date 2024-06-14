"""Base classes for trajopt solvers."""

import abc

from mlrp_course.trajopt.trajopt_problem import (
    TrajOptState,
    TrajOptTraj,
    UnconstrainedTrajOptProblem,
)
from typing import TypeAlias, Any

_TrajOptSolution: TypeAlias = Any  # intermediate representation of a solution

class UnconstrainedTrajOptSolver(abc.ABC):
    """A solver for an unconstrained trajectory optimization problem."""

    def __init__(self, warm_start: bool = True) -> None:
        self._warm_start = warm_start
        self._last_solution: TrajOptTraj | None = None

    def reset(self) -> None:
        """Reset the memory."""
        self._last_solution = None

    def solve(
        self,
        problem: UnconstrainedTrajOptProblem,
        initial_state: TrajOptState | None = None,
        horizon: int | None = None,
    ) -> TrajOptTraj:
        """Get the most recent state and return an action."""
        if initial_state is None:
            initial_state = problem.initial_state
        if horizon is None:
            horizon = problem.horizon
        solution = self._solve(problem, initial_state, horizon)
        self._last_solution = solution
        traj = self._solution_to_trajectory(solution, problem, initial_state, horizon)
        return traj

    @abc.abstractmethod
    def _solve(
        self,
        problem: UnconstrainedTrajOptProblem,
        initial_state: TrajOptState,
        horizon: int,
    ) -> _TrajOptSolution:
        """The main logic for the solver."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def _solution_to_trajectory(self, solution: _TrajOptSolution,        problem: UnconstrainedTrajOptProblem,
        initial_state: TrajOptState,
        horizon: int,
) -> TrajOptTraj:
        """Different solvers have different intermediate representations."""
        raise NotImplementedError
