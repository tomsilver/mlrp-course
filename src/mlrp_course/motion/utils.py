"""Utilities for motion planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem, RobotConf
from mlrp_course.structs import Hyperparameters
from mlrp_course.utils import (
    Trajectory,
    TrajectorySegment,
    concatenate_trajectories,
    iter_traj_with_max_distance,
)


@dataclass(frozen=True)
class MotionPlanningHyperparameters(Hyperparameters):
    """Common hyperparameters for motion planning."""

    max_velocity: float = 1.0
    collision_check_max_distance: float = 1.0
    num_shortcut_attempts: int = 100


def try_direct_path_motion_plan(
    initial_configuration: RobotConf,
    goal_configuration: RobotConf,
    has_collision: Callable[[RobotConf], bool],
    hyperparameters: MotionPlanningHyperparameters | None = None,
) -> Trajectory[RobotConf] | None:
    """Attempt to construct a trajectory directly from start to goal.

    If none is found, returns None.
    """
    if hyperparameters is None:
        hyperparameters = MotionPlanningHyperparameters()
    traj = TrajectorySegment.from_max_velocity(
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
    traj: Trajectory[RobotConf],
    rng: np.random.Generator,
    mpp: MotionPlanningProblem,
    hyperparameters: MotionPlanningHyperparameters,
) -> Trajectory[RobotConf]:
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
        traj = concatenate_trajectories([traj[:start_t], direct_path, traj[end_t:]])
    return traj
