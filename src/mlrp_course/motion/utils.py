"""Utilities for motion planning."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import Callable, Generic, Iterator, List, Sequence

import numpy as np
from spatialmath import SE2

from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem, RobotConf
from mlrp_course.structs import Hyperparameters


@dataclass(frozen=True)
class MotionPlanningHyperparameters(Hyperparameters):
    """Common hyperparameters for motion planning."""

    max_velocity: float = 1.0
    collision_check_max_distance: float = 1.0
    num_shortcut_attempts: int = 100


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

    def __getitem__(self, key: float | slice):
        """Shorthand for indexing or sub-trajectory creation."""
        if isinstance(key, float):
            return self(key)
        assert isinstance(key, slice)
        assert key.step is None
        start = key.start or 0
        end = key.stop or self.duration
        return self.get_sub_trajectory(start, end)

    @abc.abstractmethod
    def get_sub_trajectory(
        self, start_time: float, end_time: float
    ) -> RobotConfTraj[RobotConf]:
        """Create a new trajectory with time re-indexed."""


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

    def get_sub_trajectory(
        self, start_time: float, end_time: float
    ) -> RobotConfTraj[RobotConf]:
        elapsed_time = end_time - start_time
        frac = elapsed_time / self.duration
        new_duration = frac * self.duration
        return RobotConfSegment(self(start_time), self(end_time), new_duration)


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
        # Avoid numerical issues.
        time = np.clip(time, 0, self.duration)
        start_time = 0.0
        for traj in self.trajs:
            end_time = start_time + traj.duration
            if time <= end_time:
                assert time >= start_time
                return traj(time - start_time)
            start_time = end_time
        raise ValueError(f"Time {time} exceeds duration {self.duration}")

    def get_sub_trajectory(
        self, start_time: float, end_time: float
    ) -> RobotConfTraj[RobotConf]:
        new_trajs = []
        st = 0.0
        keep_traj = False
        for traj in self.trajs:
            et = st + traj.duration
            # Start keeping trajectories.
            if st <= start_time <= et:
                keep_traj = True
                # Shorten the current trajectory so it starts at start_time.
                traj = traj.get_sub_trajectory(start_time - st, traj.duration)
                st = start_time
            # Stop keeping trajectories.
            if st <= end_time <= et:
                # Shorten the current trajectory so it ends at end_time.
                traj = traj.get_sub_trajectory(0, end_time - st)
                # Finish.
                assert keep_traj
                new_trajs.append(traj)
                break
            if keep_traj:
                new_trajs.append(traj)
            st = et
        return concatenate_robot_conf_trajectories(new_trajs)


def concatenate_robot_conf_trajectories(
    trajectories: Sequence[RobotConfTraj[RobotConf]],
) -> RobotConfTraj[RobotConf]:
    """Concatenate one or more robot conf trajectories."""
    inner_trajs: List[RobotConfTraj[RobotConf]] = []
    for traj in trajectories:
        if isinstance(traj, ConcatRobotConfTraj):
            inner_trajs.extend(traj.trajs)
        else:
            inner_trajs.append(traj)
    return ConcatRobotConfTraj(inner_trajs)


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
    num_steps = int(np.ceil(traj.distance / max_distance)) + 1
    ts = np.linspace(0, traj.duration, num=num_steps, endpoint=True)
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
    return concatenate_robot_conf_trajectories(segments)


def try_direct_path_motion_plan(
    initial_configuration: RobotConf,
    goal_configuration: RobotConf,
    has_collision: Callable[[RobotConf], bool],
    hyperparameters: MotionPlanningHyperparameters | None = None,
) -> RobotConfTraj[RobotConf] | None:
    """Attempt to construct a trajectory directly from start to goal.

    If none is found, returns None.
    """
    if hyperparameters is None:
        hyperparameters = MotionPlanningHyperparameters()
    traj = RobotConfSegment.from_max_velocity(
        initial_configuration, goal_configuration, hyperparameters.max_velocity
    )
    for waypoint in iter_traj_with_max_distance(
        traj,
        hyperparameters.collision_check_max_distance,
    ):
        if has_collision(waypoint):
            return None
    return traj


def find_trajectory_shortcuts(
    traj: RobotConfTraj[RobotConf],
    rng: np.random.Generator,
    mpp: MotionPlanningProblem,
    hyperparameters: MotionPlanningHyperparameters,
) -> RobotConfTraj[RobotConf]:
    """Repeatedly attempt to find shortcuts to improve a given trajectory."""
    for _ in range(hyperparameters.num_shortcut_attempts):
        start_t, end_t = sorted(rng.uniform(0, traj.duration, size=2))
        start_conf, end_conf = traj(start_t), traj(end_t)
        # Check if direct path from start to end is collision-free.
        direct_path = try_direct_path_motion_plan(
            start_conf, end_conf, mpp.has_collision, hyperparameters
        )
        if direct_path is None:
            continue
        # Direct path works, so update the trajectory.
        traj = concatenate_robot_conf_trajectories(
            [traj[:start_t], direct_path, traj[end_t:]]
        )
    return traj
