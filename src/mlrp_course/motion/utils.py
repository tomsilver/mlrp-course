"""Utilities for motion planning."""

import abc
from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import Generic, Iterator, Sequence

import numpy as np
from spatialmath import SE2

from mlrp_course.motion.motion_planning_problem import RobotConf


class RobotConfTraj(Generic[RobotConf]):
    """A continuous-time trajectory in robot configuration space."""

    @property
    @abc.abstractmethod
    def duration(self) -> float:
        """The length of the trajectory."""

    @abc.abstractmethod
    def __call__(self, time: float) -> RobotConf:
        """Get the configuration at the given time."""

    def iter(self, max_dt: float) -> Iterator[RobotConf]:
        """Generate confs on the trajector that are at most dt apart."""
        t = 0.0
        while t < self.duration:
            yield self(t)
            t += max_dt
        # NOTE: important to iter this separately because the distance between
        # the penultimate conf and this final conf might be less than dt.
        yield self(self.duration)


@dataclass(frozen=True)
class RobotConfSegment(RobotConfTraj[RobotConf]):
    """A trajectory defined by a single start and end point."""

    start: RobotConf
    end: RobotConf
    velocity: float

    @cached_property
    def duration(self) -> float:
        dist = get_robot_conf_distance(self.start, self.end)
        return dist / self.velocity

    def __call__(self, time: float) -> RobotConf:
        assert 0 <= time <= self.duration
        s = time / self.duration
        return interpolate_robot_conf(self.start, self.end, s)


@dataclass(frozen=True)
class ConcatRobotConfTraj(RobotConfTraj[RobotConf]):
    """A trajectory that concatenates other trajectories."""

    trajs: Sequence[RobotConfTraj[RobotConf]]

    @cached_property
    def duration(self) -> float:
        return sum(t.duration for t in self.trajs)

    def __call__(self, time: float) -> RobotConf:
        start_time = 0.0
        for traj in self.trajs:
            end_time = start_time + traj.duration
            if time <= end_time:
                assert time >= start_time
                return traj(time - start_time)
            start_time = end_time
        raise ValueError(f"Time {time} exceeds duration {self.duration}")


@singledispatch
def interpolate_robot_conf(start: RobotConf, end: RobotConf, s: float) -> RobotConf:
    """Get a point on the interpolated path between start and end.

    The argument is a value between 0 and 1.
    """
    raise NotImplementedError


@interpolate_robot_conf.register
def _(start: SE2, end: SE2, s: float) -> SE2:
    return start.interp(end, s)


@singledispatch
def get_robot_conf_distance(start: RobotConf, end: RobotConf) -> float:
    """Get the distance between two robot configurations."""
    raise NotImplementedError


@get_robot_conf_distance.register
def _(start: SE2, end: SE2) -> float:
    # Many choices are possible. Here we take the maximum of the translation
    # distance and a scaled-down angular distance.
    angular_scale = 0.1
    difference = start.inv() * end
    assert isinstance(difference, SE2)
    translate_distance = np.sqrt(difference.x**2 + difference.y**2)
    angular_distance = angular_scale * abs(difference.theta())
    return max(translate_distance, angular_distance)
