"""Utilities for motion planning."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import Generic, Iterator, List, Sequence

import numpy as np
from spatialmath import SE2

from mlrp_course.motion.motion_planning_problem import RobotConf
from mlrp_course.structs import Hyperparameters


@dataclass(frozen=True)
class MotionPlanningHyperparameters(Hyperparameters):
    """Common hyperparameters for motion planning."""

    max_velocity: float = 1.0
    collision_check_max_distance: float = 1.0


class RobotConfTraj(Generic[RobotConf]):
    """A continuous-time trajectory in robot configuration space."""

    @property
    @abc.abstractmethod
    def duration(self) -> float:
        """The length of the trajectory in time."""

    @property
    @abc.abstractmethod
    def distance(self) -> float:
        """The length of the trajectory in distance."""

    @abc.abstractmethod
    def __call__(self, time: float) -> RobotConf:
        """Get the configuration at the given time."""


@dataclass(frozen=True)
class RobotConfSegment(RobotConfTraj[RobotConf]):
    """A trajectory defined by a single start and end point."""

    start: RobotConf
    end: RobotConf
    _duration: float = 1.0

    @classmethod
    def from_max_velocity(
        cls, start: RobotConf, end: RobotConf, max_velocity: float
    ) -> RobotConfSegment:
        """Create a segment from a given max velocity."""
        distance = get_robot_conf_distance(start, end)
        duration = distance / max_velocity
        return RobotConfSegment(start, end, duration)

    @cached_property
    def duration(self) -> float:
        return self._duration

    @cached_property
    def distance(self) -> float:
        return get_robot_conf_distance(self.start, self.end)

    def __call__(self, time: float) -> RobotConf:
        # Avoid numerical issues.
        time = np.clip(time, 0, self.duration)
        s = time / self.duration
        return interpolate_robot_conf(self.start, self.end, s)


@dataclass(frozen=True)
class ConcatRobotConfTraj(RobotConfTraj[RobotConf]):
    """A trajectory that concatenates other trajectories."""

    trajs: Sequence[RobotConfTraj[RobotConf]]

    @cached_property
    def duration(self) -> float:
        return sum(t.duration for t in self.trajs)

    @cached_property
    def distance(self) -> float:
        return sum(t.distance for t in self.trajs)

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


def iter_traj_with_max_distance(
    traj: RobotConfTraj[RobotConf],
    max_distance: float,
    include_start: bool = True,
    include_end: bool = True,
) -> Iterator[RobotConf]:
    """Iterate through the trajectory while guaranteeing that the distance in
    each step is no more than the given max distance."""
    num_steps = int(np.ceil(traj.distance / max_distance))
    ts = np.linspace(0, traj.duration, num=num_steps)
    if not include_start:
        ts = ts[1:]
    if not include_end:
        ts = ts[:-1]
    for t in ts:
        yield traj(t)


def robot_conf_sequence_to_trajectory(
    conf_sequence: List[RobotConf], max_velocity: float
) -> RobotConfTraj[RobotConf]:
    """Convert a sequence of motion planning nodes to a trajectory."""
    segments = []
    for t in range(len(conf_sequence) - 1):
        seg = RobotConfSegment.from_max_velocity(
            conf_sequence[t], conf_sequence[t + 1], max_velocity
        )
        segments.append(seg)
    return ConcatRobotConfTraj(segments)
