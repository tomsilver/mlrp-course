"""Tests for pddl_problem.py."""

import numpy as np
from relational_structs import GroundOperator

from mlrp_course.classical.envs.pddl_problem import PDDLPlanningProblem
from mlrp_course.utils import load_pddl_asset


def test_pddl_problem():
    """Tests for pddl_problem.py."""
    domain_str = load_pddl_asset("blocks/domain.pddl")
    problem_str = load_pddl_asset("blocks/problem1.pddl")
    problem = PDDLPlanningProblem.from_strings(domain_str, problem_str)
    action = next(iter(problem.action_space))
    assert isinstance(action, GroundOperator)
    s = problem.initial_state
    assert not problem.check_goal(s)
    applicable_actions = {a for a in problem.action_space if problem.initiable(s, a)}
    assert len(applicable_actions) > 0
    a = sorted(applicable_actions)[0]
    ns = problem.get_next_state(s, a)
    assert np.isclose(problem.get_cost(s, a, ns), 1.0)
