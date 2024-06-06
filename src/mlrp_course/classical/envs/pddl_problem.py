"""A classical planning problem defined with PDDL."""

from __future__ import annotations

from typing import Callable, FrozenSet, Set, TypeAlias

from relational_structs import GroundAtom, GroundOperator, PDDLDomain, PDDLProblem
from relational_structs.utils import all_ground_operators

from mlrp_course.classical.envs.classical_problem import ClassicalPlanningProblem
from mlrp_course.structs import Image

PDDLState: TypeAlias = FrozenSet[GroundAtom]
PDDLAction: TypeAlias = GroundOperator
PDDLGoal: TypeAlias = FrozenSet[GroundAtom]


class PDDLPlanningProblem(ClassicalPlanningProblem[PDDLState, PDDLAction]):
    """A classical planning problem defined with PDDL."""

    def __init__(
        self,
        pddl_domain: PDDLDomain,
        pddl_problem: PDDLProblem,
        render_fn: Callable[[PDDLState, PDDLGoal], Image] | None = None,
        ground_operators: Set[GroundOperator] | None = None,
    ) -> None:
        # Expose to the heuristics.
        self.pddl_domain = pddl_domain
        self.pddl_problem = pddl_problem
        self.goal_atoms = frozenset(self.pddl_problem.goal)
        if ground_operators is None:
            ground_operators = all_ground_operators(
                self.pddl_domain.operators, self.pddl_problem.objects
            )
        self.ground_operators = ground_operators
        self._render_fn = render_fn

    @classmethod
    def from_strings(
        cls,
        pddl_domain_str: str,
        pddl_problem_str: str,
        render_fn: Callable[[PDDLState, PDDLGoal], Image] | None = None,
    ) -> PDDLPlanningProblem:
        """Create a PDDLPlanningProblem from PDDL strings."""
        pddl_domain = PDDLDomain.parse(pddl_domain_str)
        pddl_problem = PDDLProblem.parse(pddl_problem_str, pddl_domain)
        return PDDLPlanningProblem(pddl_domain, pddl_problem, render_fn)

    @property
    def state_space(self) -> Set[PDDLState]:
        raise NotImplementedError("PDDL state spaces cannot be enumerated")

    @property
    def action_space(self) -> Set[PDDLAction]:
        return self.ground_operators

    @property
    def initial_state(self) -> PDDLState:
        return frozenset(self.pddl_problem.init_atoms)

    def initiable(self, state: PDDLState, action: PDDLAction) -> bool:
        return action.preconditions.issubset(state)

    def check_goal(self, state: PDDLState) -> bool:
        return self.goal_atoms.issubset(state)

    def get_cost(
        self, state: PDDLState, action: PDDLAction, next_state: PDDLState
    ) -> float:
        return 1.0

    def get_next_state(self, state: PDDLState, action: PDDLAction) -> PDDLState:
        return (state - action.delete_effects) | action.add_effects

    def render_state(self, state: PDDLState) -> Image:
        if self._render_fn is None:
            raise NotImplementedError
        goal = PDDLGoal(self.pddl_problem.goal)
        return self._render_fn(state, goal)
