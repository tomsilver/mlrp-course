"""Tests for prm.py."""

import numpy as np
from spatialmath import SE2
from tomsgeoms2d.structs import Circle, Rectangle

from mlrp_course.motion.algorithms.prm import run_prm
from mlrp_course.motion.envs.geom2d_problem import Geom2DMotionPlanningProblem
from mlrp_course.utils import TrajectorySegment


def test_prm():
    """Tests for prm.py."""
    # First test a very simple problem where a straight line should work.
    world_x_bounds = (0, 5)
    world_y_bounds = (0, 5)
    robot_init_geom = Rectangle.from_center(1, 1, 1, 1, np.pi / 4)
    robot_goal = SE2(4, 4, 0)
    obstacle_geoms = set()
    problem = Geom2DMotionPlanningProblem(
        world_x_bounds,
        world_y_bounds,
        robot_init_geom,
        robot_goal,
        obstacle_geoms,
        seed=123,
    )
    rng = np.random.default_rng(123)
    solution = run_prm(problem, rng)
    assert solution is not None
    # It should be a straight line.
    assert isinstance(solution, TrajectorySegment)

    # Test a case with obstacles, where a straight line will not work.
    obstacle_geoms = {
        Circle(2.5, 2.5, 1.0),
        Rectangle.from_center(0.5, 4, 1, 2, 0),
    }
    problem = Geom2DMotionPlanningProblem(
        world_x_bounds,
        world_y_bounds,
        robot_init_geom,
        robot_goal,
        obstacle_geoms,
        seed=123,
    )
    rng = np.random.default_rng(123)
    solution = run_prm(problem, rng)
    assert solution is not None
    # It should not be a straight line.
    assert not isinstance(solution, TrajectorySegment)
