"""Motion planning problems defined with geom2d."""

from typing import Any, ClassVar, Collection, Dict, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from spatialmath import SE2
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem
from mlrp_course.structs import Image
from mlrp_course.utils import fig2data


class SE2Space(gym.Space[SE2]):
    """A space of SE2 poses."""

    def __init__(
        self,
        rng: np.random.Generator,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        theta_bounds: Tuple[float, float] | None = None,
    ) -> None:
        if theta_bounds is None:
            theta_bounds = (-np.pi, np.pi)
        assert -np.pi <= theta_bounds[0] <= np.pi
        assert -np.pi <= theta_bounds[1] <= np.pi
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds
        self._theta_bounds = theta_bounds
        super().__init__(shape=None, dtype=None, seed=rng)

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
            and self._theta_bounds[0] <= x.theta() < self._theta_bounds[1]
        )

    @property
    def is_np_flattenable(self) -> bool:
        return False


def _geom_to_se2_pose(geom: Geom2D) -> SE2:
    if isinstance(geom, Circle):
        return SE2(geom.x, geom.y, 0.0)
    if isinstance(geom, Rectangle):
        return SE2(geom.center[0], geom.center[1], geom.theta)
    raise NotImplementedError


def _copy_geom_with_pose(geom: Geom2D, configuration: SE2) -> Geom2D:
    if isinstance(geom, Circle):
        return Circle(configuration.x, configuration.y, geom.radius)
    if isinstance(geom, Rectangle):
        return Rectangle.from_center(
            configuration.x,
            configuration.y,
            geom.width,
            geom.height,
            configuration.theta(),
        )
    raise NotImplementedError


class Geom2DMotionPlanningProblem(MotionPlanningProblem[SE2]):
    """A motion planning problem defined with geom2d."""

    render_dpi: ClassVar[int] = 150
    obstacle_render_kwargs: ClassVar[Dict[str, Any]] = {
        "fc": "gray",
        "ec": "black",
    }
    robot_current_render_kwargs: ClassVar[Dict[str, Any]] = {
        "fc": "blue",
        "ec": "black",
    }
    robot_goal_render_kwargs: ClassVar[Dict[str, Any]] = {
        "fc": (0, 1, 0, 0.5),
        "ec": "black",
        "linestyle": "dashed",
    }

    def __init__(
        self,
        world_x_bounds: Tuple[float, float],
        world_y_bounds: Tuple[float, float],
        robot_init_geom: Geom2D,
        robot_goal: SE2,
        obstacle_geoms: Collection[Geom2D],
        seed: int,
    ) -> None:
        self._world_x_bounds = world_x_bounds
        self._world_y_bounds = world_y_bounds
        self._robot_init_geom = robot_init_geom
        self._robot_goal = robot_goal
        self._robot_goal_geom = _copy_geom_with_pose(robot_init_geom, robot_goal)
        self._obstacle_geoms = obstacle_geoms
        self._rng = np.random.default_rng(seed)

    @property
    def configuration_space(self) -> SE2Space:
        return SE2Space(self._rng, self._world_x_bounds, self._world_y_bounds)

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
        world_min_x, world_max_x = self._world_x_bounds
        world_min_y, world_max_y = self._world_y_bounds

        figsize = (
            world_max_x - world_min_x,
            world_max_y - world_min_y,
        )
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=self.render_dpi)

        for obstacle_geom in self._obstacle_geoms:
            obstacle_geom.plot(ax, **self.obstacle_render_kwargs)
        self._robot_goal_geom.plot(ax, **self.robot_goal_render_kwargs)
        if configuration is None:
            robot_geom = self._robot_init_geom
        else:
            robot_geom = _copy_geom_with_pose(self._robot_init_geom, configuration)
        robot_geom.plot(ax, **self.robot_current_render_kwargs)

        pad_x = (world_max_x - world_min_x) / 25
        pad_y = (world_max_y - world_min_y) / 25
        ax.set_xlim(world_min_x - pad_x, world_max_x + pad_x)
        ax.set_ylim(world_min_y - pad_y, world_max_y + pad_y)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        img = fig2data(fig)
        plt.close()
        return img
