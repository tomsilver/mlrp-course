"""Motion planning problems defined with geom2d."""

from typing import Any, Collection, Tuple

import gymnasium as gym
import numpy as np
from spatialmath import SE2
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem
from mlrp_course.structs import Image


class SE2Space(gym.Space[SE2]):
    """A space of SE2 poses."""

    def __init__(
        self,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        theta_bounds: Tuple[float, float] | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if theta_bounds is None:
            theta_bounds = (-np.pi, np.pi)
        assert -np.pi <= theta_bounds[0] <= np.pi
        assert -np.pi <= theta_bounds[1] <= np.pi
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds
        self._theta_bounds = theta_bounds
        super().__init__(shape=None, dtype=None, seed=seed)

    def sample(self, mask: Any | None = None) -> SE2:
        assert mask is None
        x = self._np_random.uniform(self._x_bounds[0], self._x_bounds[1])
        y = self._np_random.uniform(self._y_bounds[0], self._y_bounds[1])
        theta = self._np_random.uniform(self._theta_bounds[0], self._theta_bounds[1])
        return SE2(x, y, theta)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, SE2):
            return False
        return (
            self._x_bounds[0] <= x.x < self._x_bounds[1]
            and self._y_bounds[0] <= x.y < self._y_bounds[1]
            and self._theta_bounds[0] <= x.theta < self._theta_bounds[1]
        )

    @property
    def is_np_flattenable(self) -> bool:
        return False


def _geom_to_se2_pose(geom: Geom2D) -> SE2:
    if isinstance(geom, Circle):
        return SE2(geom.x, geom.y, 0.0)
    if isinstance(geom, Rectangle):
        return SE2(geom.x, geom.y, geom.theta)
    raise NotImplementedError


def _copy_geom_with_pose(geom: Geom2D, configuration: SE2) -> Geom2D:
    if isinstance(geom, Circle):
        return Circle(configuration.x, configuration.y, geom.radius)
    if isinstance(geom, Rectangle):
        return Rectangle(
            configuration.x,
            configuration.y,
            geom.width,
            geom.height,
            configuration.theta,
        )
    raise NotImplementedError


class Geom2DMotionPlanningProblem(MotionPlanningProblem[SE2]):
    """A motion planning problem defined with geom2d."""

    def __init__(
        self,
        world_x_bounds: Tuple[float, float],
        world_y_bounds: Tuple[float, float],
        robot_init_geom: Geom2D,
        robot_goal: SE2,
        obstacle_geoms: Collection[Geom2D],
    ) -> None:
        self._world_x_bounds = world_x_bounds
        self._world_y_bounds = world_y_bounds
        self._robot_init_geom = robot_init_geom
        self._robot_goal = robot_goal
        self._obstacle_geoms = obstacle_geoms

    @property
    def configuration_space(self) -> SE2Space:
        return SE2Space(self._world_x_bounds, self._world_y_bounds)

    @property
    def initial_configuration(self) -> SE2:
        return _geom_to_se2_pose(self._robot_init_geom)

    @property
    def goal_configuration(self) -> SE2:
        return self._robot_goal

    def has_collision(self, configuration: SE2) -> bool:
        robot_geom = _copy_geom_with_pose(self._robot_init_geom, configuration)
        for obstacle_geom in self._obstacle_geoms:
            if geom2ds_intersect(robot_geom, obstacle_geom):
                return True
        return False

    def render(self, configuration: SE2 | None = None) -> Image:
        # TODO
        raise NotImplementedError
