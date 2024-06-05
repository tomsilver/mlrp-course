"""A classical planning problem defined with PDDL."""

from dataclasses import dataclass
from typing import Callable, FrozenSet, Set, TypeAlias

from relational_structs import GroundAtom, GroundOperator, PDDLDomain, PDDLProblem
from relational_structs.utils import all_ground_operators

from mlrp_course.classical.envs.classical_problem import ClassicalPlanningProblem
from mlrp_course.structs import Image

PDDLState: TypeAlias = FrozenSet[GroundAtom]
PDDLAction: TypeAlias = GroundOperator


@dataclass(frozen=True)
class PDDLGoal:
    """A goal for a PDDL planning problem."""

    goal_atoms: FrozenSet[GroundAtom]

    def __call__(self, state: PDDLState) -> bool:
        return self.goal_atoms.issubset(state)


class PDDLPlanningProblem(ClassicalPlanningProblem[PDDLState, PDDLAction]):
    """A classical planning problem defined with PDDL."""

    def __init__(
        self,
        pddl_domain_str: str,
        pddl_problem_str: str,
        render_fn: Callable[[PDDLState, PDDLGoal], Image] | None = None,
    ) -> None:
        self._pddl_domain = PDDLDomain.parse(pddl_domain_str)
        self._pddl_problem = PDDLProblem.parse(pddl_problem_str, self._pddl_domain)
        self._render_fn = render_fn
        self._all_ground_operators = all_ground_operators(
            self._pddl_domain.operators, self._pddl_problem.objects
        )

    @property
    def state_space(self) -> Set[PDDLState]:
        raise NotImplementedError("PDDL state spaces cannot be enumerated")

    @property
    def action_space(self) -> Set[PDDLAction]:
        return self._all_ground_operators

    @property
    def initial_state(self) -> PDDLState:
        return frozenset(self._pddl_problem.init_atoms)

    def initiable(self, state: PDDLState, action: PDDLAction) -> bool:
        return action.preconditions.issubset(state)

    def get_cost(
        self, state: PDDLState, action: PDDLAction, next_state: PDDLState
    ) -> float:
        return 1.0

    def get_next_state(self, state: PDDLState, action: PDDLAction) -> PDDLState:
        return (state - action.delete_effects) | action.add_effects

    def render_state(self, state: PDDLState) -> Image:
        if self._render_fn is None:
            raise NotImplementedError
        goal = PDDLGoal(self._pddl_problem.goal)
        return self._render_fn(state, goal)
