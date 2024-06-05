"""Heuristics for classical planning problems."""

import abc
from typing import Generic, TypeVar

from mlrp_course.classical.envs.classical_problem import (
    ClassicalPlanningProblem,
    DiscreteAction,
    DiscreteState,
)
from mlrp_course.classical.envs.pddl_problem import (
    PDDLAction,
    PDDLPlanningProblem,
    PDDLState,
)

_S = TypeVar("_S", bound=DiscreteState)
_A = TypeVar("_A", bound=DiscreteAction)


class ClassicalPlanningHeuristic(Generic[_S, _A]):
    """An approximator of cost-to-go."""

    def __init__(self, problem: ClassicalPlanningProblem[_S, _A]) -> None:
        self._problem = problem

    def __call__(self, state: _S) -> float:
        return self._get_cost_to_go(state)

    @abc.abstractmethod
    def _get_cost_to_go(self, state: _S) -> float:
        """Estimate the cost-to-go from this state."""


class TrivialHeuristic(ClassicalPlanningHeuristic):
    """Always estimate a cost of 0."""

    def _get_cost_to_go(self, state: _S) -> float:
        return 0.0


class PDDLHeuristic(ClassicalPlanningHeuristic[PDDLState, PDDLAction], abc.ABC):
    """An approximator of cost-to-go for PDDL Problems."""

    def __init__(self, problem: PDDLPlanningProblem) -> None:
        super().__init__(problem)
        self._goal = problem.goal_atoms


class GoalCountHeuristic(PDDLHeuristic):
    """Count the number of goal atoms not satisfied in a PDDL problem.."""

    def _get_cost_to_go(self, state: PDDLState) -> float:
        return len(self._goal - state)
