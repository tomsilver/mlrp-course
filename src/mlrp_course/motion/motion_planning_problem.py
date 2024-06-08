"""A general motion planning problem."""

import abc
from typing import Generic, TypeVar

import gymnasium as gym

from mlrp_course.structs import Image

RobotConf = TypeVar("RobotConf")  # robot configuration


class MotionPlanningProblem(Generic[RobotConf]):
    """A general motion planning problem."""

    @property
    @abc.abstractmethod
    def configuration_space(self) -> gym.Space[RobotConf]:
        """A configuration space that can be sampled."""

    @property
    @abc.abstractmethod
    def initial_configuration(self) -> RobotConf:
        """The initial robot configuration."""

    @property
    @abc.abstractmethod
    def goal_configuration(self) -> RobotConf:
        """The goal robot configuration."""

    @abc.abstractmethod
    def has_collision(self, configuration: RobotConf) -> bool:
        """Collision checking."""

    @abc.abstractmethod
    def render(self, configuration: RobotConf | None = None) -> Image:
        """Optional rendering function."""
