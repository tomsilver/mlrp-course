"""Heuristics for classical planning problems."""

import abc
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Generic, Set, TypeVar

from relational_structs import GroundAtom, GroundOperator, PDDLDomain, PDDLProblem
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

    def _construct_pddl_problem_from_state(
        self,
        state: PDDLState,
        domain: PDDLDomain | None = None,
        ground_operators: Set[GroundOperator] | None = None,
    ) -> PDDLPlanningProblem:
        """Helper function to override the initial state."""
        assert isinstance(self._problem, PDDLPlanningProblem)
        if domain is None:
            domain = self._problem.pddl_domain
        if ground_operators is None:
            ground_operators = self._problem.ground_operators
        new_pddl_problem = PDDLProblem(
            self._problem.pddl_problem.domain_name,
            self._problem.pddl_problem.problem_name,
            self._problem.pddl_problem.objects,
            state,
            self._problem.pddl_problem.goal,
        )
        return PDDLPlanningProblem(
            domain,
            new_pddl_problem,
            ground_operators=ground_operators,
        )


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

    def _construct_relaxed_problem(self, state: PDDLState) -> PDDLPlanningProblem:
        return self._construct_pddl_problem_from_state(
            state,
            domain=self._relaxed_domain,
            ground_operators=self._relaxed_ground_operators,
        )

    def _get_cost_to_go(self, state: PDDLState) -> float:
        # Run astar with goal-count to get a plan in the (relaxed) problem.
        new_problem = self._construct_relaxed_problem(state)
        _, actions, _ = run_astar(new_problem, GoalCountHeuristic(new_problem))
        return len(actions)


@dataclass
class _RPGLevel:
    """A level in a relaxed planning graph."""

    atoms: Set[GroundAtom]  # all atoms that are reachable at this level
    operators: Set[GroundOperator]  # all operators that appear executable


def _build_relaxed_planning_graph_levels(
    init_atoms: Set[GroundAtom], goal: Set[GroundAtom], operators: Set[GroundOperator]
) -> Dict[int, _RPGLevel]:
    """Helper for _RelaxedPlanningGraph."""
    # Initialize the first level (but operators will be added later).
    level = _RPGLevel(atoms=set(init_atoms), operators=set())
    t = 0
    levels = {t: level}
    # Iterate until either the goal is reached or there is no change.
    while not goal.issubset(level.atoms):
        # Add operators whose preconditions hold at this level.
        for op in operators:
            if op.preconditions.issubset(level.atoms):
                level.operators.add(op)
        # Create the next level of atoms.
        next_level = _RPGLevel(atoms=set(level.atoms), operators=set())
        for op in level.operators:
            next_level.atoms.update(op.add_effects)
        # Check for convergence.
        if level.atoms == next_level.atoms:
            break
        # Not yet converged.
        t += 1
        level = next_level
        levels[t] = level
    return levels


class _RelaxedPlanningGraph:
    """Used by certain PDDL heuristics."""

    def __init__(self, problem: PDDLPlanningProblem) -> None:
        self._problem = problem
        self._levels = _build_relaxed_planning_graph_levels(
            set(problem.initial_state),
            set(problem.goal_atoms),
            set(problem.ground_operators),
        )

    @property
    def num_levels(self) -> int:
        """The number of levels in the RPG."""
        return len(self._levels)

    @lru_cache(maxsize=None)
    def atom_to_level(self, atom: GroundAtom) -> int | None:
        """None means infinity."""
        for t in sorted(self._levels):
            level = self._levels[t]
            if atom in level.atoms:
                return t
        return None

    @lru_cache(maxsize=None)
    def operator_to_level(self, operator: GroundOperator) -> int | None:
        """None means infinity."""
        for t in sorted(self._levels):
            level = self._levels[t]
            if operator in level.operators:
                return t
        return None

    @lru_cache(maxsize=None)
    def level_to_operators(self, level: int) -> Set[GroundOperator]:
        """Get all operators at this level."""
        return {
            o
            for o in self._problem.ground_operators
            if self.operator_to_level(o) == level
        }


class HFFHeuristic(PDDLHeuristic):
    """The HFF heuristic for PDDL problems."""

    def _get_cost_to_go(self, state: PDDLState) -> float:
        problem = self._construct_pddl_problem_from_state(state)
        rpg = _RelaxedPlanningGraph(problem)
        # Determine the max level that we need to consider, and check for
        # unreachability. Also nitialize the goal levels.
        max_level = 0
        goal_levels: Dict[int, Set[GroundAtom]] = defaultdict(set)
        for goal_atom in problem.goal_atoms:
            level = rpg.atom_to_level(goal_atom)
            if level is None:
                return float("inf")  # unreachable
            max_level = max(max_level, level)
            goal_levels[level].add(goal_atom)
        # Backward pass.
        num_selected_actions = 0
        for level in range(max_level, 0, -1):
            for goal_atom in goal_levels[level]:
                # Determine which operator could have added this atom.
                candidates = {
                    o
                    for o in rpg.level_to_operators(level - 1)
                    if goal_atom in o.add_effects
                }
                assert candidates
                operator = min(candidates)  # arbitrary tie-breaking
                # Add this operator to the relaxed plan (implicitly).
                num_selected_actions += 1
                # Add the preconditions of the operator as subgoals.
                for pre in operator.preconditions:
                    pre_level = rpg.atom_to_level(pre)
                    assert pre_level is not None and pre_level < level
                    goal_levels[pre_level].add(pre)
        return num_selected_actions
