"""Heuristics for classical planning problems."""

import abc
from typing import Generic, TypeVar

from relational_structs import PDDLProblem
from relational_structs.utils import all_ground_operators

from mlrp_course.classical.algorithms.search import run_astar
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
from mlrp_course.classical.utils import delete_relax_pddl_domain

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


class DeleteRelaxationHeuristic(PDDLHeuristic):
    """Plan in the delete-relaxed version of the problem."""

    def __init__(self, problem: PDDLPlanningProblem) -> None:
        super().__init__(problem)
        # Relax the domain.
        self._relaxed_domain = delete_relax_pddl_domain(problem.pddl_domain)
        self._relaxed_ground_operators = all_ground_operators(
            self._relaxed_domain.operators, problem.pddl_problem.objects
        )

    def _get_cost_to_go(self, state: PDDLState) -> float:
        assert isinstance(self._problem, PDDLPlanningProblem)
        new_pddl_problem = PDDLProblem(
            self._problem.pddl_problem.domain_name,
            self._problem.pddl_problem.problem_name,
            self._problem.pddl_problem.objects,
            state,
            self._problem.pddl_problem.goal,
        )
        new_problem = PDDLPlanningProblem(
            self._relaxed_domain,
            new_pddl_problem,
            ground_operators=self._relaxed_ground_operators,
        )
        # Run astar with goal-count to get a plan in the (relaxed) problem.
        _, actions, _ = run_astar(new_problem, GoalCountHeuristic(new_problem))
        return len(actions)
