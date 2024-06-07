"""Tests for geom2d_problem.py."""

import numpy as np
from spatialmath import SE2
from tomsgeoms2d.structs import Circle, Rectangle

from mlrp_course.motion.envs.geom2d_problem import Geom2DMotionPlanningProblem


def test_geom2d_problem():
    """Tests for geom2d_problem.py."""
    world_x_bounds = (0, 5)
    world_y_bounds = (0, 5)
    robot_init_geom = Rectangle.from_center(1, 1, 1, 1, np.pi / 4)
    robot_goal = SE2(4, 4, 0)
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
    )
    assert not problem.has_collision(problem.initial_configuration)
    assert not problem.has_collision(robot_goal)
    assert problem.has_collision(SE2(2.5, 2.5, 1.0))
    cspace = problem.configuration_space
    cspace.seed(123)
    for _ in range(10):
        c = cspace.sample()
        assert cspace.contains(c)
        # Uncomment to render.
        # import imageio.v2 as iio
        # img = problem.render(configuration=c)
        # iio.imsave("test_geom2d_problem.png", img)
        # import ipdb; ipdb.set_trace()
