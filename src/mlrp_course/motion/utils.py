"""Utilities for motion planning."""

import abc
from dataclasses import dataclass
from functools import singledispatch
from typing import Generic, Sequence

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


@dataclass
class RobotConfSegment(RobotConfTraj[RobotConf]):
    """A trajectory defined by a single start and end point."""

    start: RobotConf
    end: RobotConf
    _duration: float

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, time: float) -> RobotConf:
        assert 0 <= time <= self.duration
        s = time / self.duration
        return interpolate_robot_conf(self.start, self.end, s)


@dataclass
class ConcatRobotConfTraj(RobotConfTraj[RobotConf]):
    """A trajectory that concatenates other trajectories."""

    trajs: Sequence[RobotConfTraj[RobotConf]]

    @property
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
