"""A general motion planning problem."""

import abc
from typing import Generic, TypeVar

import gymnasium as gym

from mlrp_course.structs import Image

_Configuration = TypeVar("_Configuration")  # robot configuration


class MotionPlanningProblem(Generic[_Configuration]):
    """A general motion planning problem."""

    @property
    @abc.abstractmethod
    def configuration_space(self) -> gym.Space[_Configuration]:
        """A configuration space that can be sampled."""

    @property
    @abc.abstractmethod
    def initial_configuration(self) -> _Configuration:
        """The initial robot configuration."""

    @property
    @abc.abstractmethod
    def goal_configuration(self) -> _Configuration:
        """The goal robot configuration."""

    @abc.abstractmethod
    def has_collision(self, configuration: _Configuration) -> bool:
        """Collision checking."""

    @abc.abstractmethod
    def render(self, configuration: _Configuration | None = None) -> Image:
        """Optional rendering function."""
